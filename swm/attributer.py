from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments

from .data import PolyMarketData
from .dataset import PriorAttributerDataset
from .utils.regressor import LLMRegressor, LLMRegressorConfig


class KLDivergenceTrainer(Trainer):
    def _process_group(
        self, model, inputs, weights, group_indices, is_prediction=False
    ):
        group_size = len(group_indices)
        chunk_size = 4
        collected_logits = []

        group_weights = weights[group_indices]
        p_dist = group_weights / (group_weights.sum() + 1e-8)
        p_dist = p_dist.to(model.device)

        for i in range(0, group_size, chunk_size):
            chunk_indices = group_indices[i : i + chunk_size]
            chunk_inputs = {
                'input_ids': inputs['input_ids'][chunk_indices],
                'attention_mask': inputs['attention_mask'][chunk_indices],
            }

            with torch.set_grad_enabled(not is_prediction):
                chunk_outputs = model(**chunk_inputs)
                chunk_logits = (
                    chunk_outputs
                    if isinstance(chunk_outputs, torch.Tensor)
                    else chunk_outputs.logits
                    if hasattr(chunk_outputs, 'logits')
                    else chunk_outputs[0]
                )

                if chunk_logits.dim() == 2 and chunk_logits.size(-1) == 1:
                    chunk_logits = chunk_logits.squeeze(-1)
                collected_logits.append(chunk_logits)

        all_logits = torch.cat(collected_logits, dim=0)
        q_dist = F.softmax(all_logits / 1.0, dim=0)

        epsilon = 1e-8
        q_dist = q_dist + epsilon
        q_dist = q_dist / q_dist.sum()

        kl_loss = torch.sum(p_dist * torch.log(p_dist / q_dist))

        return (kl_loss, q_dist, p_dist) if is_prediction else kl_loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        weights = inputs.pop('weights').to(model.device)
        group_ids = inputs.pop('group_ids').to(model.device)

        total_loss = 0.0
        num_valid_groups = 0

        for group in torch.unique(group_ids):
            group_indices = torch.where(group_ids == group)[0]
            if len(group_indices) < 2:
                continue

            loss = self._process_group(model, inputs, weights, group_indices)
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss
                num_valid_groups += 1

        avg_loss = total_loss / max(num_valid_groups, 1)
        return (avg_loss, None) if return_outputs else avg_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        weights = inputs.pop('weights').to(model.device)
        group_ids = inputs.pop('group_ids').to(model.device)

        all_losses, all_preds, all_labels = [], [], []

        for group in torch.unique(group_ids):
            group_indices = torch.where(group_ids == group)[0]
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

        return (
            torch.stack(all_losses).mean(),
            torch.stack(all_preds),
            torch.stack(all_labels),
        )


class BasicPriorAttributer:
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
        def collate_fn(batch):
            all_input_ids, all_attention_masks = [], []
            all_weights, all_group_ids = [], []
            all_market_ids, all_event_ids, all_ts, all_news = [], [], [], []

            max_len = min(
                max(item['input_ids'].size(-1) for item in batch), self.max_seq_length
            )

            for group_idx, item in enumerate(batch):
                input_ids_padded = torch.nn.functional.pad(
                    item['input_ids'][:, :max_len],
                    (0, max_len - min(item['input_ids'].size(-1), max_len)),
                    value=self.tokenizer.pad_token_id,
                )
                attention_masks = (
                    input_ids_padded != self.tokenizer.pad_token_id
                ).long()

                all_input_ids.append(input_ids_padded)
                all_attention_masks.append(attention_masks)
                all_weights.append(item['p_dist'])
                all_group_ids.append(
                    torch.full((item['p_dist'].size(0),), group_idx, dtype=torch.long)
                )
                all_market_ids.append(item['market_id'])
                all_event_ids.append(item['event_id'])
                all_ts.append(item['t'])
                all_news.append(item['news'])

            return {
                'input_ids': torch.cat(all_input_ids, dim=0),
                'attention_mask': torch.cat(all_attention_masks, dim=0),
                'weights': torch.cat(all_weights, dim=0),
                'group_ids': torch.cat(all_group_ids, dim=0),
                'market_ids': all_market_ids,
                'event_ids': all_event_ids,
                'ts': all_ts,
                'news': all_news,
            }

        return collate_fn

    def train(
        self,
        train_data: List[PolyMarketData],
        valid_data: List[PolyMarketData],
        training_args: TrainingArguments,
    ) -> str:
        """
        Train PriorAttributer using precomputed attributions.
        
        Args:
            train_data: List of markets with precomputed attributions in market.attributions
            valid_data: List of markets with precomputed attributions
            training_args: HuggingFace TrainingArguments
            
        Returns:
            Path to best model checkpoint
        """
        if self.model is None:
            self.setup_model()

        train_dataset = PriorAttributerDataset(
            markets=train_data,
            tokenizer=self.tokenizer,
            cache_dir=self.cache_dir,
        )
        valid_dataset = PriorAttributerDataset(
            markets=valid_data,
            tokenizer=self.tokenizer,
            cache_dir=self.cache_dir,
        )

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
                            p.label_ids * np.log(p.label_ids / p.predictions), axis=1
                        )
                    )
                )
            },
        )

        trainer.train()
        best_model_dir = Path(training_args.output_dir) / 'checkpoint-best'
        trainer.save_model(best_model_dir)
        return str(best_model_dir)

    def predict(
        self,
        markets: List[PolyMarketData],
        batch_size: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Predict attribution distributions for markets.
        
        Args:
            markets: List of markets with precomputed attributions in market.attributions
            batch_size: Batch size for prediction
            
        Returns:
            List of predictions with q_dist (predicted) and p_dist (ground truth)
        """
        dataset = PriorAttributerDataset(
            markets=markets,
            tokenizer=self.tokenizer,
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
            for batch in tqdm(dataloader, desc='Predicting'):
                input_ids = batch['input_ids'].to(self.model.llm.device)
                attention_mask = batch['attention_mask'].to(self.model.llm.device)
                group_ids = batch['group_ids'].to(self.model.llm.device)
                weights = batch['weights'].to(self.model.llm.device)

                group_logits_map = {}
                for group in torch.unique(group_ids):
                    group_indices = torch.where(group_ids == group)[0]
                    logits = self.model(
                        input_ids=input_ids[group_indices],
                        attention_mask=attention_mask[group_indices],
                    )
                    if logits.dim() == 2 and logits.size(-1) == 1:
                        logits = logits.squeeze(-1)
                    group_logits_map[group.item()] = logits

                for group_idx in torch.unique(group_ids):
                    group_idx = group_idx.item()
                    group_indices = torch.where(group_ids == group_idx)[0]

                    logits = group_logits_map[group_idx]
                    q_dist = F.softmax(logits, dim=0)
                    group_weights = weights[group_indices]

                    results.append(
                        {
                            'event_id': batch['event_ids'][group_idx],
                            'market_id': batch['market_ids'][group_idx],
                            't': batch['ts'][group_idx],
                            'news': batch['news'][group_idx],
                            'q_dist': q_dist.cpu().numpy().tolist(),
                            'p_dist': group_weights.cpu().numpy().tolist(),
                        }
                    )

        return results
    
    def attribute(
        self,
        timestamp: float,
        market: PolyMarketData,
    ) -> List[Dict[str, Any]]:
        """
        Generate attributions for a single market at a specific timestamp.
        
        This is used during inference with MultiEventForecaster.
        
        Args:
            timestamp: The timestamp to generate attributions for
            market: The market data
            
        Returns:
            List of {news, score} dicts
        """
        # TODO: Implement real-time attribution prediction
        # For now, return from precomputed if available
        if market.attributions:
            return market.attributions.get(str(timestamp), [])
        return []

    def save(self, path: str) -> None:
        if self.model:
            self.model.save_pretrained(path)

    def load(self, path: str) -> None:
        self.model = LLMRegressor.from_pretrained(path)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

