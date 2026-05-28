from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments

from .data import Record
from .dataset import MultiEventForecasterDataset
from .utils.regressor import LLMRegressor, LLMRegressorConfig


class WeightedTrainer(Trainer):
    """Weighted-sum trainer: one optimizer group per record.

    Each record yields N tokenized prompts (one per attributed news, or one
    no-news prompt for null records) and an N-vector of normalized weights.
    The model output is the weighted sum across prompts; loss is MSE between
    that scalar and target.p (which is the absolute price).

    The whole batch goes through a single model forward and per-group
    aggregation is done with scatter_add. One forward per step keeps the
    autograd graph identical across DDP ranks — chunked per-group forwards
    used to desync NCCL ALLREDUCE.
    """

    def __init__(self, *args, head_lr_multiplier: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_lr_multiplier = head_lr_multiplier

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        head_params, other_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'regression_head' in name:
                head_params.append(param)
            else:
                other_params.append(param)

        base_lr = self.args.learning_rate
        groups = [
            {'params': other_params, 'lr': base_lr},
            {'params': head_params, 'lr': base_lr * self.head_lr_multiplier},
        ]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        optimizer_kwargs.pop('lr', None)
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        return self.optimizer

    @staticmethod
    def _aggregate(preds, weights, group_ids, n_groups):
        """Weighted-sum per group via scatter_add."""
        group_ids = group_ids.long()
        weight_sum = torch.zeros(n_groups, device=weights.device, dtype=weights.dtype)
        weight_sum.scatter_add_(0, group_ids, weights)
        normalized = (weights / (weight_sum[group_ids] + 1e-8)).to(preds.dtype)

        acc = torch.zeros(n_groups, device=preds.device, dtype=preds.dtype)
        acc.scatter_add_(0, group_ids, preds * normalized)
        return acc

    def _forward_and_loss(self, model, inputs):
        labels = inputs.pop('labels')
        weights = inputs.pop('weights')
        group_ids = inputs.pop('group_ids')

        preds = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
        ).view(-1)
        acc_pred = self._aggregate(preds, weights, group_ids, labels.size(0))
        loss = torch.nn.functional.mse_loss(acc_pred, labels.to(acc_pred.dtype))
        return loss, acc_pred, labels.to(acc_pred.dtype)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, _, _ = self._forward_and_loss(model, inputs)
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss, acc_pred, labels = self._forward_and_loss(model, inputs)
        return (loss, acc_pred, labels)


class MultiEventForecaster:
    """Forecaster that predicts target.p from question + history + attributed news."""

    def __init__(
        self,
        model_name: str,
        max_seq_length: int = 512,
        gradient_checkpointing: bool = False,
        max_news: int = 50,
        head_lr_multiplier: float = 1.0,
        pooling_method: str = 'last_token',
        null_subsample_ratio: float = 1.0,
        window_std_threshold: float = 0.0,
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_length = max_seq_length
        self.model = None
        self.gradient_checkpointing = gradient_checkpointing
        self.max_news = max_news
        self.pooling_method = pooling_method
        self.head_lr_multiplier = head_lr_multiplier
        self.null_subsample_ratio = null_subsample_ratio
        self.window_std_threshold = window_std_threshold

    def setup_model(self) -> None:
        config = LLMRegressorConfig(
            base_model_name_or_path=self.model_name,
            max_length=self.max_seq_length,
            pooling_method=self.pooling_method,
        )
        self.model = LLMRegressor(config)
        if self.gradient_checkpointing and hasattr(self.model.llm, 'gradient_checkpointing_enable'):
            self.model.llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("Gradient checkpointing enabled (use_reentrant=False)")

    def _create_collate_fn(self):
        def collate_fn(batch):
            max_len = min(
                max(x['input_ids'].size(-1) for x in batch), self.max_seq_length
            )
            all_input_ids, all_attention_masks = [], []
            all_labels, all_weights, all_group_ids = [], [], []
            all_market_ids, all_event_ids, all_ts = [], [], []
            all_before_prices = []

            for group_idx, item in enumerate(batch):
                ids = item['input_ids'][:, :max_len]
                padded = torch.nn.functional.pad(
                    ids, (0, max_len - ids.size(-1)),
                    value=self.tokenizer.pad_token_id,
                )
                all_input_ids.append(padded)
                all_attention_masks.append((padded != self.tokenizer.pad_token_id).long())
                all_labels.append(item['label'])
                all_weights.append(item['weights'])
                all_group_ids.append(torch.full((len(item['weights']),), group_idx))
                all_market_ids.append(item['market_id'])
                all_event_ids.append(item['event_id'])
                all_ts.append(item['t'])
                all_before_prices.append(item['before_price'])

            # 'before_prices' is only consumed by predict() for the
            # pred_delta/true_delta derived fields exported to downstream
            # inference scripts; the trainer ignores it.
            return {
                'input_ids': torch.cat(all_input_ids),
                'attention_mask': torch.cat(all_attention_masks),
                'labels': torch.stack(all_labels),
                'weights': torch.cat(all_weights),
                'group_ids': torch.cat(all_group_ids),
                'before_prices': torch.stack(all_before_prices),
                'market_ids': all_market_ids,
                'event_ids': all_event_ids,
                'ts': all_ts,
            }

        return collate_fn

    def _safe_train(self, trainer):
        try:
            trainer.train()
        except FileNotFoundError:
            # HF load_best_model_at_end can race with our custom save layout;
            # the best checkpoint is still on disk.
            pass

    def _make_dataset(self, records: List[Record], null_subsample_ratio: float) -> MultiEventForecasterDataset:
        return MultiEventForecasterDataset(
            records=records,
            tokenizer=self.tokenizer,
            max_news=self.max_news,
            max_seq_length=self.max_seq_length,
            null_subsample_ratio=null_subsample_ratio,
            window_std_threshold=self.window_std_threshold,
        )

    def train(
        self,
        train_records: List[Record],
        valid_records: List[Record],
        training_args: TrainingArguments,
    ) -> str:
        if self.model is None:
            self.setup_model()

        train_dataset = self._make_dataset(train_records, self.null_subsample_ratio)
        valid_dataset = self._make_dataset(valid_records, 1.0)

        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=self._create_collate_fn(),
            head_lr_multiplier=self.head_lr_multiplier,
        )

        self._safe_train(trainer)
        if trainer.state.best_model_checkpoint:
            return trainer.state.best_model_checkpoint
        final = Path(training_args.output_dir) / 'final-model'
        trainer.save_model(final)
        return str(final)

    def predict(
        self,
        records: List[Record],
        attributer: Any = None,
        batch_size: int = 8,
        score_threshold: float = 0.0,
        top_k: int = 0,
    ) -> List[Dict[str, Union[str, float]]]:
        if attributer is not None:
            records = self._generate_attributions(
                records, attributer,
                score_threshold=score_threshold, top_k=top_k,
            )

        dataset = MultiEventForecasterDataset(
            records=records,
            tokenizer=self.tokenizer,
            max_news=self.max_news,
            max_seq_length=self.max_seq_length,
            window_std_threshold=self.window_std_threshold,
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            collate_fn=self._create_collate_fn(),
        )

        self.model.eval()
        results = []
        chunk_size = 8
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Predicting Batches'):
                input_ids = batch['input_ids'].to(self.model.llm.device)
                attention_mask = batch['attention_mask'].to(self.model.llm.device)
                weights = batch['weights'].to(self.model.llm.device)
                group_ids = batch['group_ids'].to(self.model.llm.device)
                before_prices = batch['before_prices'].to(self.model.llm.device)
                labels = batch['labels'].to(self.model.llm.device)

                for group_idx in range(len(batch['event_ids'])):
                    mask = group_ids == group_idx
                    indices = torch.where(mask)[0]
                    group_weights = weights[mask]
                    normalized = group_weights / (group_weights.sum() + 1e-8)

                    acc_pred = 0.0
                    for i in range(0, len(indices), chunk_size):
                        chunk_idx = indices[i:i + chunk_size]
                        chunk_pred = self.model(
                            input_ids=input_ids[chunk_idx],
                            attention_mask=attention_mask[chunk_idx],
                        ).view(-1)
                        acc_pred += (chunk_pred * normalized[i:i + chunk_size]).sum()

                    before_price = before_prices[group_idx].item()
                    pred_price = acc_pred.item()
                    true_price = labels[group_idx].item()

                    results.append({
                        'event_id': batch['event_ids'][group_idx],
                        'market_id': batch['market_ids'][group_idx],
                        't': batch['ts'][group_idx],
                        'pred_delta': pred_price - before_price,
                        'true_delta': true_price - before_price,
                        'pred_price': pred_price,
                        'true_price': true_price,
                        'before_price': before_price,
                    })
        return results

    def _generate_attributions(
        self,
        records: List[Record],
        attributer: Any,
        score_threshold: float = 0.0,
        top_k: int = 0,
    ) -> List[Record]:
        """Overwrite each record's attributions in place using the attributer."""
        total_kept = total_news = total_records_with_news = 0
        for record in tqdm(records, desc='Generating attributions'):
            # CRITICAL: clear any oracle attributions so a null/empty result
            # propagates as truly "no news" instead of leaking the labels.
            record.attributions = []
            news_list = record.news
            if not news_list or len(news_list) < 2:
                continue

            attrs = attributer.attribute_record(
                record,
                score_threshold=score_threshold,
                top_k=top_k,
            )
            total_news += len(news_list)
            if attrs:
                record.attributions = [
                    {'news_idx': a['news_idx'], 'score': a['score']} for a in attrs
                ]
                total_kept += len(attrs)
                total_records_with_news += 1

        if total_news > 0:
            print(f"Generated attributions for {total_records_with_news} records, "
                  f"kept {total_kept}/{total_news} news "
                  f"({total_kept / total_news * 100:.1f}%)")
        return records

    def save(self, path: str) -> None:
        if self.model:
            self.model.save_pretrained(path)

    def load(self, path: str) -> None:
        self.model = LLMRegressor.from_pretrained(path)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
