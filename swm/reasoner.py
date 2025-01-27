import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jsonlines
from transformers import TrainingArguments, AutoTokenizer

from .data import DailyNewsData, PolyMarketData
from .utils.error_handler import (
    api_calling_error_exponential_backoff,
    parsing_error_exponential_backoff,
)
from .utils.filter import TimeBasedDailyNewsFilter
from .utils.prompter import model_prompting
from .utils.utils import convert_to_date
import torch
from transformers import Trainer
import torch.nn.functional as F
from .utils.regressor import LLMRegressor, LLMRegressorConfig
from peft import LoraConfig
from .dataset import BasicPolyMarketDatasetWithEventForReasoner
from .utils.posterior_reasoner import BasicPosteriorReasoner
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


class KLDivergenceTrainer(Trainer):
    def _process_group(
        self, model, inputs, weights, group_indices, is_prediction=False
    ):
        """Process a group of items and compute KL divergence loss."""
        group_size = len(group_indices)
        chunk_size = 4
        collected_logits = []

        # Normalize weights to get proper probability distribution p
        group_weights = weights[group_indices]
        p_dist = group_weights / (group_weights.sum() + 1e-8)
        
        # Move p_dist to the same device as the model
        p_dist = p_dist.to(model.device)

        # Process in chunks to avoid OOM
        for i in range(0, group_size, chunk_size):
            chunk_indices = group_indices[i : i + chunk_size]
            chunk_inputs = {
                'input_ids': inputs['input_ids'][chunk_indices],
                'attention_mask': inputs['attention_mask'][chunk_indices],
            }

            with torch.set_grad_enabled(not is_prediction):
                # Get logits from model
                chunk_outputs = model(**chunk_inputs)
                # Handle different model output formats
                if isinstance(chunk_outputs, torch.Tensor):
                    chunk_logits = chunk_outputs
                else:
                    chunk_logits = chunk_outputs.logits if hasattr(chunk_outputs, 'logits') else chunk_outputs[0]
                
                if chunk_logits.dim() == 2 and chunk_logits.size(-1) == 1:
                    chunk_logits = chunk_logits.squeeze(-1)
                collected_logits.append(chunk_logits)

        # Combine all logits
        all_logits = torch.cat(collected_logits, dim=0)  # [group_size]
        
        # Apply temperature scaling to make logits more reasonable
        temperature = 1.0  # Adjust this parameter if needed
        scaled_logits = all_logits / temperature
        
        # Compute q distribution using softmax
        q_dist = F.softmax(scaled_logits, dim=0)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        q_dist = q_dist + epsilon
        q_dist = q_dist / q_dist.sum()  # Renormalize
        
        # Compute KL divergence loss: KL(P||Q) = sum(p_i * log(p_i/q_i))
        kl_loss = torch.sum(p_dist * torch.log(p_dist / q_dist))
        
        if is_prediction:
            return kl_loss, q_dist, p_dist
        return kl_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute average KL divergence loss across all groups."""
        weights = inputs.pop('weights')
        group_ids = inputs.pop('group_ids')
        
        # Ensure inputs are on the correct device
        weights = weights.to(model.device)
        group_ids = group_ids.to(model.device)

        total_loss = 0.0
        num_valid_groups = 0
        unique_groups = torch.unique(group_ids)

        for group in unique_groups:
            group_indices = torch.where(group_ids == group)[0]
            
            # Skip groups that are too small
            if len(group_indices) < 2:
                continue
                
            loss = self._process_group(model, inputs, weights, group_indices)
            
            # Check if loss is valid
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss
                num_valid_groups += 1

        # Compute average loss only over valid groups
        avg_loss = total_loss / max(num_valid_groups, 1)
        
        if return_outputs:
            return avg_loss, None
        return avg_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Compute predictions and losses for evaluation."""
        weights = inputs.pop('weights')
        group_ids = inputs.pop('group_ids')
        
        # Move to correct device
        weights = weights.to(model.device)
        group_ids = group_ids.to(model.device)

        all_losses = []
        all_preds = []
        all_labels = []

        unique_groups = torch.unique(group_ids)
        for group in unique_groups:
            group_indices = torch.where(group_ids == group)[0]
            
            # Skip small groups
            if len(group_indices) < 2:
                continue
                
            loss, q_dist, p_dist = self._process_group(
                model, inputs, weights, group_indices, is_prediction=True
            )
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                all_losses.append(loss.unsqueeze(0))
                all_preds.append(q_dist)
                all_labels.append(p_dist)

        if not all_losses:
            return (torch.tensor(0.0), None, None)

        final_loss = torch.stack(all_losses).mean()
        final_preds = torch.stack(all_preds)
        final_labels = torch.stack(all_labels)

        return (final_loss, final_preds, final_labels)



class BasicPriorReasoner:
    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        max_seq_length: int = 512,
        lora_config: Optional[LoraConfig] = None,
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_length = max_seq_length
        self.model = None
        self.cache_dir = Path(cache_dir)
        self.lora_config = lora_config

    def setup_model(self) -> None:
        config = LLMRegressorConfig(
            base_model_name_or_path=self.model_name, max_length=self.max_seq_length
        )
        self.model = LLMRegressor(config, lora_config=self.lora_config)

    def _create_collate_fn(self):
        """
        Similar structure to your existing code, except we do not rely on a single 'labels'
        for MSE but on 'weights' for the distribution.
        """
        def collate_fn(batch):
            # 'batch' is a list of items, each item has shape [num_items, seq_len] for input_ids, etc.
            # We'll gather them into big tensors. We track group_ids so the Trainer can separate them later.

            all_input_ids = []
            all_attention_masks = []
            all_weights = []
            all_group_ids = []
            # We'll store metadata for reference, though not used in the forward pass
            all_market_ids = []
            all_event_ids = []
            all_ts = []

            max_len = 0
            # First find a max length across groups
            for group_idx, item in enumerate(batch):
                seq_len = item['input_ids'].size(-1)
                if seq_len > max_len:
                    max_len = seq_len

            # Also clip to self.max_seq_length if needed
            max_len = min(max_len, self.max_seq_length)

            # Build final Tensors
            current_group_idx = 0
            for group_idx, item in enumerate(batch):
                # item['input_ids'] => shape [num_items, seq_len]
                # We'll pad up to max_len
                input_ids_padded = torch.nn.functional.pad(
                    item['input_ids'][:, :max_len],
                    (0, max_len - min(item['input_ids'].size(-1), max_len)),
                    value=self.tokenizer.pad_token_id,
                )
                # shape [num_items, max_len]
                attention_masks = (input_ids_padded != self.tokenizer.pad_token_id).long()

                # We'll store them in a list to cat later
                all_input_ids.append(input_ids_padded)
                all_attention_masks.append(attention_masks)

                # 'weights' => shape [num_items], the p-dist
                all_weights.append(item['p_dist'])

                # We'll keep the same 'group_idx' for all items in this group
                # so the trainer code knows which items belong together
                group_id_tensor = torch.full(
                    (item['p_dist'].size(0),),
                    group_idx,
                    dtype=torch.long
                )
                all_group_ids.append(group_id_tensor)

                all_market_ids.append(item['market_id'])
                all_event_ids.append(item['event_id'])
                all_ts.append(item['t'])

            # Finally, cat them along dimension 0
            # (Because the HF Trainer expects all samples in a single batch dimension)
            input_ids = torch.cat(all_input_ids, dim=0)
            attention_mask = torch.cat(all_attention_masks, dim=0)
            weights = torch.cat(all_weights, dim=0)
            group_ids = torch.cat(all_group_ids, dim=0)

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'weights': weights,       # Our p-dist
                'group_ids': group_ids,
                'market_ids': all_market_ids,
                'event_ids': all_event_ids,
                'ts': all_ts,
            }

        return collate_fn

    def train(
        self,
        train_data: List[PolyMarketData],
        valid_data: List[PolyMarketData],
        training_args: TrainingArguments,
        posterior_reasoner: BasicPosteriorReasoner,
    ) -> str:
        """
        Train the model using KLDivergenceTrainer, computing KL-based loss.
        """
        if self.model is None:
            self.setup_model()

        # Build your dataset. This dataset must yield items with 'input_ids', 'attention_mask', 'weights'.
        # E.g. BasicPolyMarketDatasetWithEventForPredictor that sets 'weights' as the distribution p.
        train_dataset = BasicPolyMarketDatasetWithEventForReasoner(
            markets=train_data,
            tokenizer=self.tokenizer,
            reasoner=posterior_reasoner,
            cache_dir=self.cache_dir,
        )
        valid_dataset = BasicPolyMarketDatasetWithEventForReasoner(
            markets=valid_data,
            tokenizer=self.tokenizer,
            reasoner=posterior_reasoner,
            cache_dir=self.cache_dir,
        )

        # Instead of WeightedTrainer, we'll use KLDivergenceTrainer
        trainer = KLDivergenceTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=self._create_collate_fn(),
            compute_metrics=lambda p: {
                'kl_div': float(
                    np.mean(
                        np.sum(
                            p.label_ids * np.log(p.label_ids / (p.predictions)),
                            axis=1
                        )
                    )
                )
            }
        )

        trainer.train()
        best_model_dir = Path(training_args.output_dir) / 'checkpoint-best'
        trainer.save_model(best_model_dir)
        return str(best_model_dir)

    def predict(
        self,
        markets: List[PolyMarketData],
        reasoner: BasicPosteriorReasoner,
        batch_size: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Use the trained model to produce a distribution q (softmax of logits) for each group.
        """
        # Build dataset with reasoner
        dataset = BasicPolyMarketDatasetWithEventForReasoner(
            markets=markets,
            tokenizer=self.tokenizer,
            reasoner=reasoner,
            cache_dir=self.cache_dir,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._create_collate_fn(),
        )

        self.model.eval()
        results = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Predicting Batches'):
                input_ids = batch['input_ids'].to(self.model.llm.device)
                attention_mask = batch['attention_mask'].to(self.model.llm.device)
                weights = batch['weights'].to(self.model.llm.device)
                group_ids = batch['group_ids'].to(self.model.llm.device)

                # We'll replicate the chunk logic if needed,
                # or do a simpler approach if the batch is small enough.
                # For each group in the batch, collect logits.
                unique_groups = torch.unique(group_ids)
                group_logits_map = {}

                for group in unique_groups:
                    group_indices = torch.where(group_ids == group)[0]
                    chunk_inputs = {
                        'input_ids': input_ids[group_indices],
                        'attention_mask': attention_mask[group_indices],
                    }
                    logits = self.model(**chunk_inputs)
                    # shape [num_items_in_group]
                    if logits.dim() == 2 and logits.size(-1) == 1:
                        logits = logits.squeeze(-1)
                    group_logits_map[group.item()] = logits

                # Then interpret them as a distribution
                for i, group_idx in enumerate(unique_groups):
                    # Convert to distribution
                    logits = group_logits_map[group_idx.item()]
                    q_dist = F.softmax(logits, dim=0)

                    # We can store or do something with it
                    # If the dataset stored metadata in the batch (like market_id, event_id),
                    # we can match them by the group index.
                    results.append(
                        {
                            'event_id': batch['event_ids'][group_idx],
                            'market_id': batch['market_ids'][group_idx],
                            't': batch['ts'][group_idx],
                            'q_dist': q_dist.cpu().numpy().tolist(),
                        }
                    )
        return results

    def save(self, path: str) -> None:
        if self.model:
            self.model.save_pretrained(path)

    def load(self, path: str) -> None:
        self.model = LLMRegressor.from_pretrained(path)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')