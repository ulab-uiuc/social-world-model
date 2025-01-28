from pathlib import Path
from typing import Dict, List, Optional

import torch
from peft import LoraConfig
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments

from .data import PolyMarketData
from .dataset import BasicPolyMarketDatasetWithEventForPredictor
from .reasoner import BasicPriorReasoner
from .utils.posterior_reasoner import BasicPosteriorReasoner
from .utils.regressor import LLMRegressor, LLMRegressorConfig


class WeightedTrainer(Trainer):
    def _process_group(
        self, model, inputs, weights, group_indices, labels, is_prediction=False
    ):
        group_size = len(group_indices)
        chunk_size = 8
        acc_pred = 0

        group_weights = weights[group_indices]
        normalized_weights = group_weights / group_weights.sum()

        for i in range(0, group_size, chunk_size):
            chunk_indices = group_indices[i : i + chunk_size]
            chunk_inputs = {
                'input_ids': inputs['input_ids'][chunk_indices],
                'attention_mask': inputs['attention_mask'][chunk_indices],
            }
            chunk_weights = normalized_weights[i : i + chunk_size]

            with (
                torch.amp.autocast('cuda', enabled=self.args.fp16)
                if not is_prediction
                else torch.no_grad()
            ):
                chunk_preds = model(**chunk_inputs)
                chunk_preds = chunk_preds.view(-1)
                acc_pred += (chunk_preds * chunk_weights).sum()

        group_label = labels[group_indices[0]]
        loss = torch.nn.functional.mse_loss(acc_pred, group_label)

        return loss, acc_pred, group_label

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop('labels')
        weights = inputs.pop('weights')
        group_ids = inputs.pop('group_ids')

        total_loss = 0
        num_valid_groups = 0
        for group in torch.unique(group_ids):
            group_indices = torch.where(group_ids == group)[0]
            loss, _, _ = self._process_group(
                model, inputs, weights, group_indices, labels
            )
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss
                num_valid_groups += 1

        return total_loss / max(num_valid_groups, 1)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.pop('labels')
        weights = inputs.pop('weights')
        group_ids = inputs.pop('group_ids')

        all_losses, all_preds, all_labels = [], [], []
        for group in torch.unique(group_ids):
            group_indices = torch.where(group_ids == group)[0]
            loss, pred, label = self._process_group(
                model, inputs, weights, group_indices, labels, is_prediction=True
            )

            if not torch.isnan(loss) and not torch.isinf(loss):
                all_losses.append(loss)
                all_preds.append(pred)
                all_labels.append(label)

        return (
            torch.stack(all_losses).mean(),
            torch.stack(all_preds),
            torch.stack(all_labels),
        )


class BasicPredictor:
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
            max_len = max(
                max(x['input_ids'].size(-1) for x in batch), self.max_seq_length
            )

            all_input_ids = []
            all_attention_masks = []
            all_labels = []
            all_weights = []
            all_group_ids = []
            all_market_ids = []
            all_event_ids = []
            all_ts = []

            for group_idx, item in enumerate(batch):
                padded_inputs = torch.nn.functional.pad(
                    item['input_ids'],
                    (0, max_len - item['input_ids'].size(-1)),
                    value=self.tokenizer.pad_token_id,
                )
                all_input_ids.append(padded_inputs)
                mask = (padded_inputs != self.tokenizer.pad_token_id).long()
                all_attention_masks.append(mask)
                all_labels.append(item['label'])
                all_weights.append(item['weights'])
                all_group_ids.append(torch.full((len(item['weights']),), group_idx))
                all_market_ids.append(item['market_id'])
                all_event_ids.append(item['event_id'])
                all_ts.append(item['t'])

            return {
                'input_ids': torch.cat(all_input_ids),
                'attention_mask': torch.cat(all_attention_masks),
                'labels': torch.cat(all_labels),
                'weights': torch.cat(all_weights),
                'group_ids': torch.cat(all_group_ids),
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
        reasoner: BasicPosteriorReasoner,
    ) -> str:
        if self.model is None:
            self.setup_model()

        train_dataset = BasicPolyMarketDatasetWithEventForPredictor(
            markets=train_data,
            tokenizer=self.tokenizer,
            reasoner=reasoner,
            cache_dir=self.cache_dir,
        )
        valid_dataset = BasicPolyMarketDatasetWithEventForPredictor(
            markets=valid_data,
            tokenizer=self.tokenizer,
            reasoner=reasoner,
            cache_dir=self.cache_dir,
        )

        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=self._create_collate_fn(),
            compute_metrics=lambda p: {
                'mse': mean_squared_error(p.label_ids, p.predictions)
            },
        )

        trainer.train()
        best_model_dir = Path(training_args.output_dir) / 'checkpoint-best'
        trainer.save_model(best_model_dir)
        return str(best_model_dir)

    def predict(
        self,
        markets: List[PolyMarketData],
        reasoner: BasicPriorReasoner,
        batch_size: int = 8,
    ) -> Dict[str, Dict[str, float]]:
        dataset = BasicPolyMarketDatasetWithEventForPredictor(
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
                labels = batch['labels'].to(self.model.llm.device)
                weights = batch['weights'].to(self.model.llm.device)
                group_ids = batch['group_ids'].to(self.model.llm.device)

                for group_idx in range(
                    len(batch['event_ids'])
                ):  # Iterate over actual groups
                    group_mask = group_ids == group_idx
                    group_inputs = {
                        'input_ids': input_ids[group_mask],
                        'attention_mask': attention_mask[group_mask],
                    }

                    group_preds = self.model(**group_inputs).view(-1)

                    group_weights = weights[group_mask]
                    group_weights = group_weights / group_weights.sum()

                    weighted_pred = (group_preds * group_weights).sum()

                    results.append(
                        {
                            'event_id': batch['event_ids'][group_idx],
                            'market_id': batch['market_ids'][group_idx],
                            't': batch['ts'][group_idx],
                            'prediction': weighted_pred.item(),
                            'ground_truth': labels[group_mask][0].item(),
                        }
                    )
        return results

    def save(self, path: str) -> None:
        if self.model:
            self.model.save_pretrained(path)

    def load(self, path: str) -> None:
        self.model = LLMRegressor.from_pretrained(path)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
