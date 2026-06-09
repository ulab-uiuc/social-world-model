from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments

from .data import Record
from .dataset import MultiEventForecasterDataset, collate_padded_groups
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

    def __init__(
        self,
        *args,
        head_lr_multiplier: float = 1.0,
        per_news_loss: bool = False,
        odds_null_categorical: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.odds_null_categorical = odds_null_categorical
        self.head_lr_multiplier = head_lr_multiplier
        self.per_news_loss = per_news_loss

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        base_lr = self.args.learning_rate
        decay = self.args.weight_decay
        # Four groups: {head, backbone} x {decay, no-decay}. Biases and 1-D
        # params (LayerNorm/RMSNorm weights) get no weight decay, matching HF's
        # default split — the old code applied decay to all params indiscriminately.
        groups = {
            'head_decay': {
                'params': [],
                'lr': base_lr * self.head_lr_multiplier,
                'weight_decay': decay,
            },
            'head_nodecay': {
                'params': [],
                'lr': base_lr * self.head_lr_multiplier,
                'weight_decay': 0.0,
            },
            'other_decay': {'params': [], 'lr': base_lr, 'weight_decay': decay},
            'other_nodecay': {'params': [], 'lr': base_lr, 'weight_decay': 0.0},
        }
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_head = 'regression_head' in name
            no_decay = param.ndim <= 1 or name.endswith('.bias')
            key = ('head' if is_head else 'other') + (
                '_nodecay' if no_decay else '_decay'
            )
            groups[key]['params'].append(param)

        group_list = [g for g in groups.values() if g['params']]
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        optimizer_kwargs.pop('lr', None)
        optimizer_kwargs.pop('weight_decay', None)  # set per-group above
        self.optimizer = optimizer_cls(group_list, **optimizer_kwargs)
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
        labels = labels.to(acc_pred.dtype)
        if self.per_news_loss:
            # Responsibility-weighted (mixture-of-experts) loss matching the paper's
            # L_wm = E_{Z~pi}[-log P_theta(.|e^Z)]: score EACH news against the move,
            # then weight by its attribution -- vs the default "error of the weighted
            # mean". The expectation is inside the per-news squared error, so a
            # candidate can't hide behind the average -> every attributed news must
            # itself explain the shift. Removes the averaging ceiling / mean-collapse
            # escape route.
            gid = group_ids.long()
            if self.odds_null_categorical:
                # weights are already the odds categorical pi_i (sum 1-pi_0); the
                # missing pi_0 mass = no-news predicting 0 (a constant, no gradient).
                # Do NOT renormalize to 1 -- that would discard the null mass.
                norm_w = weights.to(preds.dtype)
            else:
                wsum = torch.zeros(
                    labels.size(0), device=weights.device, dtype=weights.dtype
                )
                wsum.scatter_add_(0, gid, weights)
                norm_w = (weights / (wsum[gid] + 1e-8)).to(preds.dtype)
            lbl_pn = labels[gid]
            se = (preds - lbl_pn) ** 2  # per-news squared error
            inner = torch.zeros(labels.size(0), device=preds.device, dtype=preds.dtype)
            inner.scatter_add_(
                0, gid, norm_w * se
            )  # Sum_i pi_i (mu_i - delta)^2 per record
            loss = inner.mean()
            return loss, acc_pred, labels
        loss = torch.nn.functional.mse_loss(acc_pred, labels)
        return loss, acc_pred, labels

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
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
        predict_delta: bool = True,
        per_news_loss: bool = False,
        odds_null_categorical: bool = False,
        null_rho0: float = 1.0,
        odds_eps: float = 1e-3,
        odds_temp: float = 1.0,
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
        self.predict_delta = predict_delta
        self.per_news_loss = per_news_loss
        # odds+null categorical: convert independent per-news Bernoulli scores into
        # a joint (k+1) categorical with a null prior mass rho0. News weights sum to
        # 1-pi_0 (no-news contributes 0); loss/predict must NOT renormalize.
        self.odds_null_categorical = odds_null_categorical
        self.null_rho0 = null_rho0
        self.odds_eps = odds_eps
        self.odds_temp = odds_temp

    def setup_model(self) -> None:
        config = LLMRegressorConfig(
            base_model_name_or_path=self.model_name,
            max_length=self.max_seq_length,
            pooling_method=self.pooling_method,
            max_news=self.max_news,
            predict_delta=self.predict_delta,
        )
        self.model = LLMRegressor(config)
        # Full fine-tune: backbone loads in bf16 but the regression head is
        # fp32, and FSDP refuses to flatten mixed-dtype params. Cast the whole
        # model to fp32 (uniform) so FSDP can wrap it; TrainingArguments bf16
        # then drives FSDP mixed-precision (bf16 compute, fp32 master weights).
        self.model = self.model.float()
        if self.gradient_checkpointing and hasattr(
            self.model.llm, 'gradient_checkpointing_enable'
        ):
            self.model.llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False}
            )
            print('Gradient checkpointing enabled (use_reentrant=False)')

    def _create_collate_fn(self):
        def collate_fn(batch):
            out = collate_padded_groups(
                batch,
                self.tokenizer.pad_token_id,
                self.max_seq_length,
            )
            out['labels'] = torch.stack([x['label'] for x in batch])
            out['weights'] = torch.cat([x['weights'] for x in batch])
            # 'before_prices' is only consumed by predict() for the
            # pred_delta/true_delta derived fields; the trainer ignores it.
            out['before_prices'] = torch.stack([x['before_price'] for x in batch])
            return out

        return collate_fn

    def _safe_train(self, trainer):
        try:
            trainer.train()
        except FileNotFoundError:
            # HF load_best_model_at_end can race with our custom save layout;
            # the best checkpoint is still on disk.
            pass

    def _make_dataset(
        self, records: List[Record], null_subsample_ratio: float
    ) -> MultiEventForecasterDataset:
        return MultiEventForecasterDataset(
            records=records,
            tokenizer=self.tokenizer,
            max_news=self.max_news,
            max_seq_length=self.max_seq_length,
            null_subsample_ratio=null_subsample_ratio,
            predict_delta=self.predict_delta,
            odds_null_categorical=self.odds_null_categorical,
            null_rho0=self.null_rho0,
            odds_eps=self.odds_eps,
            odds_temp=self.odds_temp,
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
            per_news_loss=self.per_news_loss,
            odds_null_categorical=self.odds_null_categorical,
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
        weight_temperature: float = 1.0,
    ) -> List[Dict[str, Union[str, float]]]:
        if attributer is not None:
            records = self._generate_attributions(
                records,
                attributer,
                score_threshold=score_threshold,
                top_k=top_k,
            )

        dataset = MultiEventForecasterDataset(
            records=records,
            tokenizer=self.tokenizer,
            max_news=self.max_news,
            max_seq_length=self.max_seq_length,
            predict_delta=self.predict_delta,
            include_nonews_candidate=getattr(self, 'include_nonews_candidate', False),
            odds_null_categorical=getattr(self, 'odds_null_categorical', False),
            null_rho0=getattr(self, 'null_rho0', 1.0),
            odds_eps=getattr(self, 'odds_eps', 1e-3),
            odds_temp=getattr(self, 'odds_temp', 1.0),
            direct_soft_routing=getattr(self, 'direct_soft_routing', False),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
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
                    # weight_temperature sharpens (<1) or flattens (>1) the
                    # attribution-weighted average. T<1 concentrates mass on the
                    # top news, countering dilution when many news are attributed.
                    if weight_temperature != 1.0:
                        group_weights = group_weights.clamp(min=1e-8).pow(
                            1.0 / weight_temperature
                        )
                    if getattr(self, 'odds_null_categorical', False) or getattr(
                        self, 'direct_soft_routing', False
                    ):
                        # weights are the categorical pi_i (sum 1-pi_0); use as-is
                        # so the missing pi_0 mass shrinks the aggregate (no-news -> 0).
                        normalized = group_weights
                    else:
                        normalized = group_weights / (group_weights.sum() + 1e-8)

                    acc_pred = 0.0
                    for i in range(0, len(indices), chunk_size):
                        chunk_idx = indices[i : i + chunk_size]
                        chunk_pred = self.model(
                            input_ids=input_ids[chunk_idx],
                            attention_mask=attention_mask[chunk_idx],
                        ).view(-1)
                        acc_pred += (chunk_pred * normalized[i : i + chunk_size]).sum()

                    before_price = before_prices[group_idx].item()
                    # In delta mode the model + label are deltas off before_price;
                    # in price mode they are absolute prices. Reconstruct the
                    # complementary field so downstream gets both either way.
                    if self.predict_delta:
                        pred_delta = acc_pred.item()
                        true_delta = labels[group_idx].item()
                        pred_price = before_price + pred_delta
                        true_price = before_price + true_delta
                    else:
                        pred_price = acc_pred.item()
                        true_price = labels[group_idx].item()
                        pred_delta = pred_price - before_price
                        true_delta = true_price - before_price

                    results.append(
                        {
                            'event_id': batch['event_ids'][group_idx],
                            'market_id': batch['market_ids'][group_idx],
                            't': batch['ts'][group_idx],
                            'pred_delta': pred_delta,
                            'true_delta': true_delta,
                            'pred_price': pred_price,
                            'true_price': true_price,
                            'before_price': before_price,
                        }
                    )
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
            if not news_list:
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
            print(
                f'Generated attributions for {total_records_with_news} records, '
                f'kept {total_kept}/{total_news} news '
                f'({total_kept / total_news * 100:.1f}%)'
            )
        return records

    def save(self, path: str) -> None:
        if self.model:
            self.model.save_pretrained(path)

    def load(self, path: str) -> None:
        self.model = LLMRegressor.from_pretrained(path)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        # Checkpoint config is the source of truth for must-match-training
        # hyperparameters; restore them over any inference-script defaults.
        cfg = self.model.config
        if getattr(cfg, 'pooling_method', None) is not None:
            self.pooling_method = cfg.pooling_method
        if getattr(cfg, 'max_news', None) is not None:
            self.max_news = cfg.max_news
        if getattr(cfg, 'predict_delta', None) is not None:
            self.predict_delta = cfg.predict_delta
