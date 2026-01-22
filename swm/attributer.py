from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments

from .data import MarketData
from .dataset import PriorAttributerDataset
from .utils.regressor import LLMRegressor, LLMRegressorConfig


class KLDivergenceTrainer(Trainer):
    def _process_group(
        self, model, inputs, weights, group_indices, device, is_prediction=False
    ):
        group_size = len(group_indices)
        chunk_size = 4
        collected_logits = []

        group_weights = weights[group_indices]
        p_dist = group_weights / (group_weights.sum() + 1e-8)
        p_dist = p_dist.to(device)

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

        # Add epsilon to both distributions to avoid log(0) = -inf
        # This prevents NaN when computing p * log(p/q) with p=0
        epsilon = 1e-8
        q_dist = q_dist + epsilon
        q_dist = q_dist / q_dist.sum()
        p_dist = p_dist + epsilon
        p_dist = p_dist / p_dist.sum()

        kl_loss = torch.sum(p_dist * torch.log(p_dist / q_dist))
        
        # Scale loss by group_size to compensate for small gradients when group is large
        # This ensures consistent learning rate effectiveness across different group sizes
        kl_loss = kl_loss * group_size

        return (kl_loss, q_dist, p_dist) if is_prediction else kl_loss

    def _get_device(self, model):
        """Get device from model, handling DataParallel wrapper."""
        if hasattr(model, 'device'):
            return model.device
        elif hasattr(model, 'module'):
            return next(model.module.parameters()).device
        else:
            return next(model.parameters()).device

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        device = self._get_device(model)
        weights = inputs.pop('weights').to(device)
        group_ids = inputs.pop('group_ids').to(device)

        losses = []

        for group in torch.unique(group_ids):
            group_indices = torch.where(group_ids == group)[0]
            if len(group_indices) < 2:
                continue

            loss = self._process_group(model, inputs, weights, group_indices, device)
            if not torch.isnan(loss) and not torch.isinf(loss):
                losses.append(loss)

        # Ensure we always return a tensor
        if losses:
            avg_loss = torch.stack(losses).mean()
        else:
            # Return zero tensor with grad if no valid groups
            avg_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return (avg_loss, None) if return_outputs else avg_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        device = self._get_device(model)
        weights = inputs.pop('weights').to(device)
        group_ids = inputs.pop('group_ids').to(device)

        all_losses, all_preds, all_labels = [], [], []

        for group in torch.unique(group_ids):
            group_indices = torch.where(group_ids == group)[0]
            if len(group_indices) < 2:
                continue

            loss, q_dist, p_dist = self._process_group(
                model, inputs, weights, group_indices, device, is_prediction=True
            )

            if not torch.isnan(loss) and not torch.isinf(loss):
                all_losses.append(loss.unsqueeze(0))
                all_preds.append(q_dist)
                all_labels.append(p_dist)

        if not all_losses:
            return (torch.tensor(0.0, device=device), None, None)

        # Return only loss - can't stack preds/labels as they have different sizes per group
        # The eval loss is the main metric we care about
        avg_loss = torch.stack(all_losses).mean()
        return (avg_loss, None, None)


class BasicPriorAttributer:
    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        max_seq_length: int = 512,
        lora_config: Optional[LoraConfig] = None,
        gradient_checkpointing: bool = False,
        max_news_per_bp: int = 50,
        target_temperature: float = 0.5,  # Lower = sharper target distribution
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_length = max_seq_length
        self.model = None
        self.cache_dir = Path(cache_dir)
        self.lora_config = lora_config
        self.gradient_checkpointing = gradient_checkpointing
        self.max_news_per_bp = max_news_per_bp
        self.target_temperature = target_temperature

    def setup_model(self) -> None:
        config = LLMRegressorConfig(
            base_model_name_or_path=self.model_name, max_length=self.max_seq_length
        )
        self.model = LLMRegressor(config, lora_config=self.lora_config)
        
        # Enable gradient checkpointing to save memory
        if self.gradient_checkpointing:
            if hasattr(self.model.llm, 'gradient_checkpointing_enable'):
                # use_reentrant=False is required for inputs without requires_grad
                self.model.llm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                print("Gradient checkpointing enabled (use_reentrant=False)")

    def _create_collate_fn(self):
        def collate_fn(batch):
            all_input_ids, all_attention_masks = [], []
            all_weights, all_group_ids = [], []
            all_market_ids, all_event_ids, all_ts = [], [], []

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

            return {
                'input_ids': torch.cat(all_input_ids, dim=0),
                'attention_mask': torch.cat(all_attention_masks, dim=0),
                'weights': torch.cat(all_weights, dim=0),
                'group_ids': torch.cat(all_group_ids, dim=0),
                'market_ids': all_market_ids,
                'event_ids': all_event_ids,
                'ts': all_ts,
            }

        return collate_fn

    def train(
        self,
        train_data: List[MarketData],
        valid_data: List[MarketData],
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
            max_news_per_bp=self.max_news_per_bp,
            target_temperature=self.target_temperature,
        )
        valid_dataset = PriorAttributerDataset(
            markets=valid_data,
            tokenizer=self.tokenizer,
            cache_dir=self.cache_dir,
            max_news_per_bp=self.max_news_per_bp,
            target_temperature=self.target_temperature,
        )

        def compute_metrics(eval_pred):
            """Compute evaluation metrics for KL training."""
            preds, labels = eval_pred
            # Since prediction_step returns None for preds/labels (can't stack different sizes),
            # we can't compute detailed metrics here. The eval_loss is the main metric.
            if preds is None or labels is None:
                return {}
            
            # preds and labels are flattened distributions
            # Compute KL divergence: sum(p * log(p/q))
            epsilon = 1e-8
            preds = np.clip(preds, epsilon, 1.0)
            labels = np.clip(labels, epsilon, 1.0)
            
            # KL divergence
            kl_div = np.mean(labels * np.log(labels / preds))
            
            # Top-1 accuracy: does the highest predicted match highest true?
            if len(preds) > 0 and len(labels) > 0:
                pred_top = np.argmax(preds)
                label_top = np.argmax(labels)
                top1_acc = float(pred_top == label_top)
            else:
                top1_acc = 0.0
            
            return {
                'kl_div': float(kl_div),
                'top1_acc': top1_acc,
            }
        
        trainer = KLDivergenceTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=self._create_collate_fn(),
            compute_metrics=compute_metrics,
        )

        trainer.train()
        best_model_dir = Path(training_args.output_dir) / 'checkpoint-best'
        trainer.save_model(best_model_dir)
        return str(best_model_dir)

    def predict(
        self,
        markets: List[MarketData],
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
                            'q_dist': q_dist.cpu().numpy().tolist(),
                            'p_dist': group_weights.cpu().numpy().tolist(),
                        }
                    )

        return results
    
    def attribute_breakpoint(
        self,
        market: MarketData,
        breakpoint: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Generate attributions for a single breakpoint using the trained model.
        
        This is the real-time inference method used during forecaster prediction.
        
        Args:
            market: The market data
            breakpoint: A breakpoint dict with 'news', 'window_history', 'after' fields
            
        Returns:
            List of {news_idx, score, news} dicts with attribution scores
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")
        
        news_list = breakpoint.get('news', [])
        if not news_list or len(news_list) < 2:
            return []
        
        window_history = breakpoint.get('window_history', [])
        target = breakpoint.get('after', {})
        
        # Build prompts for each news item (same as training)
        all_input_ids = []
        all_attention_masks = []
        
        for news_item in news_list[:self.max_news_per_bp]:
            prompt = self._build_prompt_with_news(market, window_history, target, news_item)
            encoding = self.tokenizer(
                prompt,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt',
            )
            all_input_ids.append(encoding['input_ids'])
            all_attention_masks.append(encoding['attention_mask'])
        
        # Stack and move to device
        input_ids = torch.cat(all_input_ids, dim=0).to(self.model.llm.device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(self.model.llm.device)
        
        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            if logits.dim() == 2 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            
            # Convert to probability distribution
            scores = F.softmax(logits, dim=0)
        
        # Build attribution results
        attributions = []
        for idx, (news_item, score) in enumerate(zip(news_list[:self.max_news_per_bp], scores)):
            attributions.append({
                'news_idx': idx,
                'score': score.item(),
                'news': news_item,
            })
        
        return attributions
    
    def _build_prompt_with_news(
        self,
        market: MarketData,
        window_history: List[Dict],
        target: Dict,
        news_item: Dict,
    ) -> str:
        """Build prompt for attribution prediction (same as training)."""
        from .utils.utils import unix_to_date
        
        lines = [f'Prediction Market: {market.question}']
        if market.description:
            desc = market.description[:200] + '...' if len(market.description or '') > 200 else market.description
            lines.append(f'Description: {desc}')
        
        # Show recent price history (BEFORE the breakpoint)
        target_ts = target.get('t')
        history_before_target = [
            day for day in window_history 
            if day.get('t') != target_ts
        ]
        
        lines.append('\nRecent price history:')
        for day in history_before_target[-5:]:
            date = unix_to_date(day['t'])
            lines.append(f"  {date}: {day['p']:.3f}")
        
        # Target date
        target_date = unix_to_date(target['t'])
        lines.append(f'\nPredicting for: {target_date}')
        
        # Add the news article
        lines.append('\nNews article:')
        news_title = news_item.get('title', '')
        news_desc = news_item.get('description', '') or ''
        news_date = news_item.get('published_at', '')
        if news_date:
            lines.append(f'Date: {news_date}')
        lines.append(f'Title: {news_title}')
        if news_desc:
            desc = news_desc[:300] + '...' if len(news_desc) > 300 else news_desc
            lines.append(f'Content: {desc}')
        
        lines.append('\nHow relevant is this news to the prediction market outcome?')
        
        return '\n'.join(lines)

    def save(self, path: str) -> None:
        if self.model:
            self.model.save_pretrained(path)

    def load(self, path: str) -> None:
        self.model = LLMRegressor.from_pretrained(path)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

