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


class KLDivergenceTrainer(Trainer):
    def _process_group(
        self, model, inputs, weights, group_indices, is_prediction=False
    ):
        """
        Process a single group of items that share a 'group_id'.
        
        Instead of summing predictions, we:
          1) Collect all logits in a list (or tensor).
          2) 'weights' for these items is interpreted as p_i for KL.
          3) We compute q = softmax(logits).
          4) The loss is cross_entropy = - sum_i p_i log(q_i).
        """
        group_size = len(group_indices)
        chunk_size = 4  # or whatever chunk size to avoid OOM
        collected_logits = []

        # p-dist is the target distribution.
        # If not yet normalized, we can do it here:
        group_weights = weights[group_indices]
        p_dist = group_weights / group_weights.sum()

        for i in range(0, group_size, chunk_size):
            chunk_indices = group_indices[i : i + chunk_size]
            chunk_inputs = {
                'input_ids': inputs['input_ids'][chunk_indices],
                'attention_mask': inputs['attention_mask'][chunk_indices],
            }

            with (
                torch.amp.autocast('cuda', enabled=self.args.fp16)
                if not is_prediction
                else torch.no_grad()
            ):
                chunk_logits = model(**chunk_inputs)
                # Suppose the model returns shape [batch_size] or [batch_size, 1].
                # Make it [batch_size] either way:
                if chunk_logits.dim() == 2 and chunk_logits.size(-1) == 1:
                    chunk_logits = chunk_logits.squeeze(-1)
                collected_logits.append(chunk_logits)

        # Concatenate all item logits for this group
        all_logits = torch.cat(collected_logits, dim=0)  # shape [group_size]

        # For predictions (in eval), we might want the softmax distribution or the raw logits
        if is_prediction:
            # We'll compute the cross-entropy as well for logging
            q_dist = F.softmax(all_logits, dim=0)
            ce_loss = -(p_dist * torch.log(q_dist + 1e-9)).sum()
            # Return both the cross-entropy and the predicted distribution
            return ce_loss, q_dist, p_dist
        else:
            # Training step => compute cross-entropy = -sum_i p_i log q_i
            q_dist = F.softmax(all_logits, dim=0)
            ce_loss = -(p_dist * torch.log(q_dist + 1e-9)).sum()
            return ce_loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Overrides the default HF Trainer compute_loss to handle grouping
        and compute KL-based loss for each group.
        """
        # Remove or pop out the labels if you have them, but here 'weights' is our p-dist
        weights = inputs.pop('weights')  # shape ~ [sum_of_items_in_batch]
        group_ids = inputs.pop('group_ids')  # shape ~ [sum_of_items_in_batch]

        total_loss = 0.0
        unique_groups = torch.unique(group_ids)

        for group in unique_groups:
            group_indices = torch.where(group_ids == group)[0]
            loss = self._process_group(model, inputs, weights, group_indices, is_prediction=False)
            total_loss += loss

        # Average over the number of groups in the batch
        avg_loss = total_loss / len(unique_groups)
        return avg_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        For evaluation, we do a similar approach but also return predictions
        so HF can compute metrics.
        """
        weights = inputs.pop('weights')
        group_ids = inputs.pop('group_ids')

        all_losses = []
        all_preds = []
        all_labels = []  # In KL scenario, 'labels' might be p_dist

        unique_groups = torch.unique(group_ids)
        for group in unique_groups:
            group_indices = torch.where(group_ids == group)[0]
            loss, q_dist, p_dist = self._process_group(
                model, inputs, weights, group_indices, is_prediction=True
            )
            all_losses.append(loss.unsqueeze(0))

            # We'll store q_dist as 'predictions' and p_dist as 'labels'
            # so that compute_metrics can see them
            all_preds.append(q_dist.unsqueeze(0))  # shape [1, group_size]
            all_labels.append(p_dist.unsqueeze(0)) # shape [1, group_size]

        # Combine
        final_loss = torch.cat(all_losses).mean()
        final_preds = torch.cat(all_preds, dim=0)   # shape [num_groups, group_size]
        final_labels = torch.cat(all_labels, dim=0) # shape [num_groups, group_size]

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
                # p.predictions => shape [num_groups, group_size]
                # p.label_ids   => shape [num_groups, group_size]
                # We can compute average cross-entropy or KL
                'ce': -(p.label_ids * np.log(p.predictions + 1e-9)).sum(axis=1).mean()
            },
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