from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import torch
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig

from .data import PolyMarketData
from .utils.regressor import LLMRegressor, LLMRegressorConfig
from .reasoner import BasicPosteriorReasoner, BasicPriorReasoner
from .dataset import BasicPolyMarketDatasetWithEvent


class WeightedTrainer(Trainer):
    def evaluation_loop(self, *args, **kwargs):
        output = super().evaluation_loop(*args, **kwargs)
        
        # Get weights from eval dataset
        weights = []
        eval_dataloader = self.get_eval_dataloader()
        for batch in eval_dataloader:
            weights.extend(batch['weights'].cpu().numpy())
            
        output.metrics['weighted_mse'] = mean_squared_error(
            output.predictions, 
            output.label_ids,
            sample_weight=weights
        )
        return output

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        weights = inputs.pop("weights")
        group_ids = inputs.pop("group_ids")
        
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Group predictions
        unique_groups = torch.unique(group_ids)
        loss = 0
        
        for group in unique_groups:
            mask = group_ids == group
            group_preds = logits[mask]
            group_weights = weights[mask]
            group_weights = group_weights / group_weights.sum()  # Normalize weights
            group_label = labels[mask][0]  # All labels in group are same
            
            # Weighted expectation
            expected_pred = (group_preds * group_weights).sum()
            loss += torch.nn.functional.mse_loss(expected_pred, group_label)
            
        loss = loss / len(unique_groups)
        return (loss, outputs) if return_outputs else loss


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
            max_len = max(max(x['input_ids'].size(-1) for x in batch), self.max_seq_length)
            
            all_input_ids = []
            all_attention_masks = []
            all_labels = []
            all_weights = []
            all_group_ids = []
            
            import pdb; pdb.set_trace()
            for group_idx, item in enumerate(batch):
                padded_inputs = torch.nn.functional.pad(
                    item['input_ids'],
                    (0, max_len - item['input_ids'].size(-1)),
                    value=self.tokenizer.pad_token_id
                )
                all_input_ids.append(padded_inputs)
                mask = (padded_inputs != self.tokenizer.pad_token_id).long()
                all_attention_masks.append(mask)
                all_labels.append(item['label'])
                all_weights.append(item['weights'])
                all_group_ids.append(torch.full((len(item['weights']),), group_idx))
                
            return {
                'input_ids': torch.cat(all_input_ids),
                'attention_mask': torch.cat(all_attention_masks),
                'labels': torch.cat(all_labels),
                'weights': torch.cat(all_weights),
                'group_ids': torch.cat(all_group_ids)
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

        train_dataset = BasicPolyMarketDatasetWithEvent(
            markets=train_data,
            tokenizer=self.tokenizer,
            reasoner=reasoner,
            cache_dir=self.cache_dir,
        )
        valid_dataset = BasicPolyMarketDatasetWithEvent(
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
            compute_metrics=lambda p: {'mse': mean_squared_error(p.label_ids, p.predictions)}
        )

        trainer.train()
        best_model_dir = Path(training_args.output_dir) / 'checkpoint-best'
        trainer.save_model(best_model_dir)
        return str(best_model_dir)

    def predict(
        self, 
        markets: List[PolyMarketData], 
        reasoner: BasicPriorReasoner,
        batch_size: int = 8
    ) -> Dict[str, Dict[str, float]]:
        dataset = BasicPolyMarketDatasetWithEvent(
            markets=markets, 
            tokenizer=self.tokenizer,
            reasoner=reasoner, 
            cache_dir=self.cache_dir
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._create_collate_fn(),
        )

        # Store predictions by (market_id, timestamp)
        predictions_map = defaultdict(list)
        weights_map = defaultdict(list)
        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Predicting Batches'):
                input_ids = batch['input_ids'].to(self.model.llm.device)
                attention_mask = batch['attention_mask'].to(self.model.llm.device)
                labels = batch['labels'].to(self.model.llm.device)
                weights = batch['weights'].to(self.model.llm.device)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                predictions = outputs['predictions'].view(-1).cpu()

                # Group predictions by market and timestamp
                for i in range(len(predictions)):
                    key = (batch['market_ids'][i], batch['ts'][i])
                    predictions_map[key].append(predictions[i])
                    weights_map[key].append(weights[i].cpu())

        # Calculate weighted expectations
        results = []
        for key in predictions_map:
            market_id, t = key
            preds = torch.stack(predictions_map[key])
            weights = torch.stack(weights_map[key])
            weights = weights / weights.sum()  # Normalize weights
            
            expected_pred = (preds * weights).sum().item()
            
            results.append({
                'market_id': market_id,
                't': t,
                'prediction': expected_pred,
                'ground_truth': labels[0].item(),  # All labels in group are same
            })

        return results

    def save(self, path: str) -> None:
        if self.model:
            self.model.save_pretrained(path)

    def load(self, path: str) -> None:
        self.model = LLMRegressor.from_pretrained(path)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
