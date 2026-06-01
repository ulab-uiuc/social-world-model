from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from peft import LoraConfig
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

    def __init__(self, *args, logit_temperature: float = 1.0,
                 routing_loss_weight: float = 0.0, reverse_kl: bool = False, neg_bce_weight: float = 0.0,
                 per_news_bce: bool = False, head_lr_multiplier: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.logit_temperature = logit_temperature
        self.routing_loss_weight = routing_loss_weight
        self.reverse_kl = reverse_kl
        self.neg_bce_weight = neg_bce_weight
        self.head_lr_multiplier = head_lr_multiplier
        # per-news Bernoulli mode: drop the softmax/KL entirely; each news is an
        # independent sigmoid trained with soft-target BCE to its 0-1 posterior
        # relevance. no-news is NOT a competing slot — it emerges as Π(1-p_i).
        self.per_news_bce = per_news_bce

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
            'head_decay': {'params': [], 'lr': base_lr * self.head_lr_multiplier, 'weight_decay': decay},
            'head_nodecay': {'params': [], 'lr': base_lr * self.head_lr_multiplier, 'weight_decay': 0.0},
            'other_decay': {'params': [], 'lr': base_lr, 'weight_decay': decay},
            'other_nodecay': {'params': [], 'lr': base_lr, 'weight_decay': 0.0},
        }
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            is_head = 'regression_head' in name
            no_decay = param.ndim <= 1 or name.endswith('.bias')
            key = ('head' if is_head else 'other') + ('_nodecay' if no_decay else '_decay')
            groups[key]['params'].append(param)
        group_list = [g for g in groups.values() if g['params']]
        from transformers import Trainer as _Trainer
        optimizer_cls, optimizer_kwargs = _Trainer.get_optimizer_cls_and_kwargs(self.args)
        optimizer_kwargs.pop('lr', None); optimizer_kwargs.pop('weight_decay', None)
        self.optimizer = optimizer_cls(group_list, **optimizer_kwargs)
        return self.optimizer

    def _per_news_bce_loss(self, model, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        raw = inputs.pop('raw_scores')                  # per-prompt g_i in [0,1]
        group_ids = inputs.pop('group_ids').long()
        inputs.pop('p_dist', None); inputs.pop('is_null', None)
        for k in ('market_ids', 'event_ids', 'ts'):
            inputs.pop(k, None)
        logits = model(input_ids=inputs['input_ids'],
                       attention_mask=inputs['attention_mask']).view(-1)
        # Exclude the no-news prompt (always the LAST prompt of each contiguous
        # group) — it is not trained as a slot; routing is emergent at inference.
        is_last = torch.ones_like(group_ids, dtype=torch.bool)
        is_last[:-1] = group_ids[1:] != group_ids[:-1]
        news_mask = ~is_last
        tgt = raw.to(logits.dtype).clamp(0.0, 1.0)
        # soft-target BCE = per-news Bernoulli cross-entropy (= the factorized
        # ELBO KL up to a target-only constant). Pulls gold up, pushes off-target
        # (g_i=0) down — the negative suppression forward-KL lacks — with NO
        # softmax competition, so it can't inflate the no-news signal.
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits[news_mask], tgt[news_mask])

    @staticmethod
    def _segment_softmax(logits: torch.Tensor, group_ids: torch.Tensor, n_groups: int) -> torch.Tensor:
        max_per = torch.full(
            (n_groups,), float('-inf'),
            device=logits.device, dtype=logits.dtype,
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
        is_null = inputs.pop('is_null', None)
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
        # KL(p ‖ q) = Σ p · log(p/q); contribution is 0 wherever p_i = 0.
        if getattr(self, 'reverse_kl', False):
            # reverse KL(q‖p) = Σ q·log(q/p): mode-seeking. q>0 where p≈0
            # (irrelevant news) is heavily penalized -> actively suppresses
            # off-target news and sharpens the distribution. Gradient flows
            # through q (the model softmax); p (target) is constant.
            kl_per_prompt = q * (torch.log(q.clamp(min=eps)) - torch.log(p.clamp(min=eps)))
        else:
            kl_per_prompt = p * (torch.log(p.clamp(min=eps)) - torch.log(q.clamp(min=eps)))
        kl_per_group = torch.zeros(n_groups, device=logits.device, dtype=logits.dtype)
        kl_per_group.scatter_add_(0, group_ids, kl_per_prompt)
        loss = kl_per_group.mean()

        # Routing classification: the no-news option is the LAST prompt of each
        # (contiguous) group. BCE pushes its prob -> 1 for null records, 0 for
        # has-news records, directly training the "is there causal news?" router.
        if self.routing_loss_weight > 0 and is_null is not None:
            is_last = torch.ones_like(group_ids, dtype=torch.bool)
            is_last[:-1] = group_ids[1:] != group_ids[:-1]
            q_no_news = q[is_last].clamp(eps, 1 - eps)            # (n_groups,), group order
            tgt = is_null.to(q_no_news.dtype)
            bce = -(tgt * torch.log(q_no_news) + (1 - tgt) * torch.log(1 - q_no_news)).mean()
            loss = loss + self.routing_loss_weight * bce

        # Per-news relevance BCE: directly supply the negative-suppression signal
        # that forward KL(p‖q) lacks (its p_i=0 terms vanish). For each NEWS prompt,
        # push sigmoid(logit) -> 1 if gold-attributed (p_i>0) else -> 0. The no-news
        # prompt is excluded here (handled by the routing BCE above).
        if self.neg_bce_weight > 0:
            is_last_n = torch.ones_like(group_ids, dtype=torch.bool)
            is_last_n[:-1] = group_ids[1:] != group_ids[:-1]
            news_mask = ~is_last_n
            if news_mask.any():
                rel_tgt = (p[news_mask] > 0).to(logits.dtype)
                bce_rel = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits[news_mask], rel_tgt)
                loss = loss + self.neg_bce_weight * bce_rel
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = self._per_news_bce_loss(model, inputs) if self.per_news_bce else self._kl_loss(model, inputs)
        return (loss, None) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss = self._per_news_bce_loss(model, inputs) if self.per_news_bce else self._kl_loss(model, inputs)
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
        target_sharpen: float = 1.0,
        routing_loss_weight: float = 0.0,
        reverse_kl: bool = False,
        neg_bce_weight: float = 0.0,
        per_news_bce: bool = False,
        head_lr_multiplier: float = 1.0,
        lora_config: Optional[LoraConfig] = None,
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
        self.target_sharpen = target_sharpen
        self.routing_loss_weight = routing_loss_weight
        self.reverse_kl = reverse_kl
        self.neg_bce_weight = neg_bce_weight
        self.per_news_bce = per_news_bce
        self.lora_config = lora_config
        # Two-stage null gate (set by inference). null_gate=True scores the
        # "no relevant news" option; null_gate_threshold=None means parameter-
        # free (null when no-news ranks #1), otherwise null when its softmax
        # score >= threshold.
        self.null_gate = False
        self.null_gate_threshold: Optional[float] = None

    def setup_model(self) -> None:
        config = LLMRegressorConfig(
            base_model_name_or_path=self.model_name, max_length=self.max_seq_length,
            max_news=self.max_news, target_temperature=self.target_temperature,
            per_news_bce=self.per_news_bce,
        )
        self.model = LLMRegressor(config, lora_config=self.lora_config)
        if self.lora_config is None:
            # Full fine-tune: backbone is bf16 but the head is fp32, and FSDP
            # refuses to flatten mixed-dtype params. Cast to uniform fp32; the
            # TrainingArguments bf16 flag then drives FSDP mixed precision.
            self.model = self.model.float()
        if self.gradient_checkpointing and hasattr(self.model.llm, 'gradient_checkpointing_enable'):
            self.model.llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("Gradient checkpointing enabled (use_reentrant=False)")
        if self.lora_config is not None and hasattr(self.model.llm, 'print_trainable_parameters'):
            self.model.llm.print_trainable_parameters()

    def _create_collate_fn(self):
        def collate_fn(batch):
            out = collate_padded_groups(
                batch, self.tokenizer.pad_token_id, self.max_seq_length,
            )
            out['p_dist'] = torch.cat([item['p_dist'] for item in batch], dim=0)
            if 'raw_scores' in batch[0]:
                out['raw_scores'] = torch.cat([item['raw_scores'] for item in batch], dim=0)
            out['is_null'] = torch.tensor(
                [float(item.get('is_null', False)) for item in batch], dtype=torch.float,
            )
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
            target_sharpen=self.target_sharpen,
        )
        valid_dataset = PriorAttributerDataset(
            records=valid_records,
            tokenizer=self.tokenizer,
            max_news=self.max_news,
            max_seq_length=self.max_seq_length,
            target_sharpen=self.target_sharpen,
        )

        trainer = KLDivergenceTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=self._create_collate_fn(),
            logit_temperature=self.target_temperature,
            routing_loss_weight=self.routing_loss_weight,
            reverse_kl=self.reverse_kl,
            neg_bce_weight=self.neg_bce_weight,
            per_news_bce=self.per_news_bce,
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
            dataset, batch_size=batch_size, shuffle=False,
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
                    results.append({
                        'event_id': batch['event_ids'][group_idx],
                        'market_id': batch['market_ids'][group_idx],
                        't': batch['ts'][group_idx],
                        'q_dist': q_dist.cpu().numpy().tolist(),
                        'p_dist': p_dist_batch[indices].cpu().numpy().tolist(),
                    })

        return results

    def attribute_record(
        self,
        record: Record,
        score_threshold: float = 0.0,
        top_k: int = 0,
    ) -> List[Dict[str, Any]]:
        """Score each news item for a single v6 record (real-time inference)."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")

        news_list = record.news
        if not news_list:
            return []

        target = record.target
        all_input_ids, all_attention_masks = [], []
        for news in news_list[:self.max_news]:
            prompt = build_attributer_news_prompt(record, target, news)
            enc = self.tokenizer(
                prompt, padding='max_length', truncation=True,
                max_length=self.max_seq_length, return_tensors='pt',
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
            prompt, padding='max_length', truncation=True,
            max_length=self.max_seq_length, return_tensors='pt',
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
                ids = input_ids[i:i + chunk_size].to(self.model.llm.device)
                mask = attention_mask[i:i + chunk_size].to(self.model.llm.device)
                logits = self.model(input_ids=ids, attention_mask=mask)
                if logits.dim() == 2 and logits.size(-1) == 1:
                    logits = logits.squeeze(-1)
                all_logits.append(logits.cpu())
            logits = torch.cat(all_logits, dim=0).to(self.model.llm.device)

        if getattr(self, 'per_news_bce', False):
            # Per-news Bernoulli: each news scored independently; no-news is the
            # last prompt but NOT a competing slot — it emerges as Π(1-p_i).
            p = torch.sigmoid(logits)
            news_scores = p[:n_news]
            # 1-max(p_i): robust emergent routing, invariant to #news (Π(1-p_i)
            # collapses toward 0 as candidates grow). null=>all p low=>~1; has-news=>~1-max.
            no_news_score = float(1.0 - news_scores.clamp(min=0.0, max=1.0).max().item()) if news_scores.numel() else 1.0
        else:
            scores = F.softmax(logits / self.target_temperature, dim=0)
            # Split off the no-news option (always last). news_scores keep their
            # training-consistent scale (they sum to 1 - no_news_score).
            no_news_score = scores[-1].item()
            news_scores = scores[:n_news]
        if self.null_gate:
            max_news_score = news_scores.max().item() if n_news > 0 else 0.0
            thr = self.null_gate_threshold
            is_null = (no_news_score >= thr) if thr is not None else (no_news_score >= max_news_score)
            if is_null:
                return []
        scores = news_scores

        attributions = []
        for idx, (news, score) in enumerate(zip(news_list[:self.max_news], scores)):
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
        if getattr(cfg, 'per_news_bce', None) is not None:
            self.per_news_bce = bool(cfg.per_news_bce)
