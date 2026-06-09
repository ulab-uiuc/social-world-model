from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments

from .data import Record
from .dataset import (
    PriorAttributerDataset,
    build_attributer_news_prompt,
    build_attributer_no_news_prompt,
    collate_padded_groups,
)
from .utils.regressor import LLMRegressor, LLMRegressorConfig


class KLDivergenceTrainer(Trainer):
    """Per-group KL(target ‖ softmax(logits / T)).

    Single batched forward; per-group softmax via scatter ops. One forward per
    step keeps the autograd graph identical across DDP ranks — chunked
    per-group forwards used to desync NCCL ALLREDUCE.
    """

    def __init__(
        self,
        *args,
        logit_temperature: float = 1.0,
        head_lr_multiplier: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.logit_temperature = logit_temperature
        self.head_lr_multiplier = head_lr_multiplier

    def create_optimizer(self):
        # Give the regression_head a larger LR than the backbone: with a uniform
        # LR the head under-fits and the attributer output stays much flatter
        # (eff~5) than the KL target (eff~2). A bigger head LR lets it produce the
        # logit spread needed to match the sharp posterior.
        if self.optimizer is not None or self.head_lr_multiplier == 1.0:
            return super().create_optimizer()
        base_lr = self.args.learning_rate
        decay = self.args.weight_decay
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
        from transformers import Trainer as _Trainer

        optimizer_cls, optimizer_kwargs = _Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        optimizer_kwargs.pop('lr', None)
        optimizer_kwargs.pop('weight_decay', None)
        self.optimizer = optimizer_cls(group_list, **optimizer_kwargs)
        return self.optimizer

    @staticmethod
    def _segment_softmax(
        logits: torch.Tensor, group_ids: torch.Tensor, n_groups: int
    ) -> torch.Tensor:
        max_per = torch.full(
            (n_groups,),
            float('-inf'),
            device=logits.device,
            dtype=logits.dtype,
        )
        max_per.scatter_reduce_(0, group_ids, logits, reduce='amax', include_self=True)
        max_per = torch.where(max_per.isfinite(), max_per, torch.zeros_like(max_per))
        exp_shifted = (logits - max_per[group_ids]).exp()
        sum_per = torch.zeros(n_groups, device=logits.device, dtype=logits.dtype)
        sum_per.scatter_add_(0, group_ids, exp_shifted)
        return exp_shifted / sum_per[group_ids].clamp(min=1e-12)

    def _kl_loss(self, model, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        targets = inputs.pop('p_dist')
        group_ids = inputs.pop('group_ids').long()
        for k in ('market_ids', 'event_ids', 'ts'):
            inputs.pop(k, None)

        logits = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
        ).view(-1)

        n_groups = int(group_ids.amax().item()) + 1
        q = self._segment_softmax(logits / self.logit_temperature, group_ids, n_groups)

        eps = 1e-8
        p = targets.to(logits.dtype).clamp(min=0.0)
        # forward KL(p ‖ q) = Σ p · log(p/q); contribution is 0 wherever p_i = 0.
        kl_per_prompt = p * (torch.log(p.clamp(min=eps)) - torch.log(q.clamp(min=eps)))
        kl_per_group = torch.zeros(n_groups, device=logits.device, dtype=logits.dtype)
        kl_per_group.scatter_add_(0, group_ids, kl_per_prompt)
        return kl_per_group.mean()

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        loss = self._kl_loss(model, inputs)
        return (loss, None) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self._kl_loss(model, inputs)
        return (loss, None, None)


class BasicPriorAttributer:
    def __init__(
        self,
        model_name: str,
        max_seq_length: int = 512,
        gradient_checkpointing: bool = False,
        max_news: int = 50,
        target_temperature: float = 0.5,
        null_subsample_ratio: float = 1.0,
        null_odds: float = 1.0,
        odds_eps: float = 1e-3,
        odds_temp: float = 1.0,
        head_lr_multiplier: float = 1.0,
    ):
        self.head_lr_multiplier = head_lr_multiplier
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_length = max_seq_length
        self.model = None
        self.gradient_checkpointing = gradient_checkpointing
        self.max_news = max_news
        self.target_temperature = target_temperature
        self.null_subsample_ratio = null_subsample_ratio
        self.null_odds = null_odds
        self.odds_eps = odds_eps
        self.odds_temp = odds_temp
        # Two-stage null gate (set by inference). null_gate=True scores the
        # "no relevant news" option; null_gate_threshold=None means parameter-
        # free (null when no-news ranks #1), otherwise null when its softmax
        # score >= threshold.
        self.null_gate = False
        self.null_gate_threshold: Optional[float] = None

    def setup_model(self) -> None:
        config = LLMRegressorConfig(
            base_model_name_or_path=self.model_name,
            max_length=self.max_seq_length,
            max_news=self.max_news,
            target_temperature=self.target_temperature,
        )
        self.model = LLMRegressor(config)
        # Full fine-tune: backbone is bf16 but the head is fp32, and FSDP
        # refuses to flatten mixed-dtype params. Cast to uniform fp32; the
        # TrainingArguments bf16 flag then drives FSDP mixed precision.
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
            out['p_dist'] = torch.cat([item['p_dist'] for item in batch], dim=0)
            return out

        return collate_fn

    def train(
        self,
        train_records: List[Record],
        valid_records: List[Record],
        training_args: TrainingArguments,
        resume_from_checkpoint: Optional[str] = None,
    ) -> str:
        if self.model is None:
            self.setup_model()

        train_dataset = PriorAttributerDataset(
            records=train_records,
            tokenizer=self.tokenizer,
            max_news=self.max_news,
            max_seq_length=self.max_seq_length,
            null_subsample_ratio=self.null_subsample_ratio,
            null_odds=self.null_odds,
            odds_eps=self.odds_eps,
            odds_temp=self.odds_temp,
        )
        valid_dataset = PriorAttributerDataset(
            records=valid_records,
            tokenizer=self.tokenizer,
            max_news=self.max_news,
            max_seq_length=self.max_seq_length,
            null_odds=self.null_odds,
            odds_eps=self.odds_eps,
            odds_temp=self.odds_temp,
        )

        trainer = KLDivergenceTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=self._create_collate_fn(),
            logit_temperature=self.target_temperature,
            head_lr_multiplier=self.head_lr_multiplier,
        )

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        if trainer.state.best_model_checkpoint:
            return trainer.state.best_model_checkpoint
        final = Path(training_args.output_dir) / 'final-model'
        trainer.save_model(final)
        return str(final)

    def predict(
        self,
        records: List[Record],
        batch_size: int = 8,
    ) -> List[Dict[str, Any]]:
        dataset = PriorAttributerDataset(
            records=records,
            tokenizer=self.tokenizer,
            max_news=self.max_news,
            max_seq_length=self.max_seq_length,
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
                p_dist_batch = batch['p_dist'].to(self.model.llm.device)

                for group in torch.unique(group_ids):
                    group_idx = group.item()
                    indices = torch.where(group_ids == group)[0]
                    logits = self.model(
                        input_ids=input_ids[indices],
                        attention_mask=attention_mask[indices],
                    )
                    if logits.dim() == 2 and logits.size(-1) == 1:
                        logits = logits.squeeze(-1)
                    # Match the training/inference temperature (training used
                    # softmax(logits / target_temperature); plain softmax here
                    # produced a different distribution than what was optimized).
                    q_dist = F.softmax(logits / self.target_temperature, dim=0)
                    results.append(
                        {
                            'event_id': batch['event_ids'][group_idx],
                            'market_id': batch['market_ids'][group_idx],
                            't': batch['ts'][group_idx],
                            'q_dist': q_dist.cpu().numpy().tolist(),
                            'p_dist': p_dist_batch[indices].cpu().numpy().tolist(),
                        }
                    )

        return results

    def attribute_record(
        self,
        record: Record,
        score_threshold: float = 0.0,
        top_k: int = 0,
    ) -> List[Dict[str, Any]]:
        """Score each news item for a single v6 record (real-time inference)."""
        if self.model is None:
            raise ValueError('Model not loaded. Call load() first.')

        news_list = record.news
        if not news_list:
            return []

        target = record.target
        all_input_ids, all_attention_masks = [], []
        for news in news_list[: self.max_news]:
            prompt = build_attributer_news_prompt(record, target, news)
            enc = self.tokenizer(
                prompt,
                padding='max_length',
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors='pt',
            )
            all_input_ids.append(enc['input_ids'])
            all_attention_masks.append(enc['attention_mask'])
        n_news = len(all_input_ids)

        # ALWAYS append the "no relevant news" option so it competes inside the
        # same softmax as the news items, exactly as training did (the dataset
        # always appends it). Dropping it when null_gate=False rescaled the news
        # scores relative to training. null_gate below only decides whether to
        # *act* on that option, not whether it participates in the softmax.
        prompt = build_attributer_no_news_prompt(record, target)
        enc = self.tokenizer(
            prompt,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt',
        )
        all_input_ids.append(enc['input_ids'])
        all_attention_masks.append(enc['attention_mask'])

        input_ids = torch.cat(all_input_ids, dim=0)
        attention_mask = torch.cat(all_attention_masks, dim=0)

        self.model.eval()
        chunk_size = 8
        all_logits = []
        with torch.no_grad():
            for i in range(0, input_ids.size(0), chunk_size):
                ids = input_ids[i : i + chunk_size].to(self.model.llm.device)
                mask = attention_mask[i : i + chunk_size].to(self.model.llm.device)
                logits = self.model(input_ids=ids, attention_mask=mask)
                if logits.dim() == 2 and logits.size(-1) == 1:
                    logits = logits.squeeze(-1)
                all_logits.append(logits.cpu())
            logits = torch.cat(all_logits, dim=0).to(self.model.llm.device)

        scores = F.softmax(logits / self.target_temperature, dim=0)
        # Split off the no-news option (always last). news_scores keep their
        # training-consistent scale (they sum to 1 - no_news_score).
        no_news_score = scores[-1].item()
        news_scores = scores[:n_news]
        if self.null_gate:
            max_news_score = news_scores.max().item() if n_news > 0 else 0.0
            thr = self.null_gate_threshold
            is_null = (
                (no_news_score >= thr)
                if thr is not None
                else (no_news_score >= max_news_score)
            )
            if is_null:
                return []
        scores = news_scores

        attributions = []
        for idx, (news, score) in enumerate(zip(news_list[: self.max_news], scores)):
            s = score.item()
            if score_threshold > 0 and s < score_threshold:
                continue
            attributions.append({'news_idx': idx, 'score': s, 'news': news})
        if top_k > 0 and len(attributions) > top_k:
            attributions.sort(key=lambda x: x['score'], reverse=True)
            attributions = attributions[:top_k]
        return attributions

    def save(self, path: str) -> None:
        if self.model:
            self.model.save_pretrained(path)

    def load(self, path: str) -> None:
        self.model = LLMRegressor.from_pretrained(path)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        # Checkpoint config is the source of truth for must-match-training
        # hyperparameters; restore them over any inference-script defaults.
        cfg = self.model.config
        if getattr(cfg, 'max_news', None) is not None:
            self.max_news = cfg.max_news
        if getattr(cfg, 'target_temperature', None) is not None:
            self.target_temperature = cfg.target_temperature
