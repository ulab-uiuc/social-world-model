# utils/regressor.py

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModel, PretrainedConfig, PreTrainedModel


class LLMRegressorConfig(PretrainedConfig):
    model_type = 'llm_regressor'

    def __init__(
        self,
        base_model_name_or_path: Optional[str] = None,
        max_length: Optional[int] = 512,
        pooling_method: Optional[str] = 'last_token',
        max_news: Optional[int] = None,
        target_temperature: Optional[float] = None,
        predict_delta: Optional[bool] = None,
        bounded_output: Optional[bool] = None,
        n_extra_features: Optional[int] = 0,
        per_news_bce: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.max_length = max_length
        self.pooling_method = pooling_method
        # Inference-critical hyperparameters that MUST match training. Persisted
        # in the checkpoint config so load() can restore them instead of relying
        # on inference-script defaults (which silently drifted from training).
        self.max_news = max_news
        self.target_temperature = target_temperature
        self.predict_delta = predict_delta
        self.bounded_output = bounded_output
        self.n_extra_features = n_extra_features or 0
        # Attributer trained with per-news Bernoulli BCE (sigmoid per news, no
        # softmax): inference must score sigmoid(logit) and derive no-news as
        # Π(1-p_i) instead of taking a softmax over (news ∪ no-news).
        self.per_news_bce = per_news_bce


class LLMRegressor(PreTrainedModel):
    config_class = LLMRegressorConfig

    def __init__(
        self,
        config: LLMRegressorConfig,
        lora_config: Optional[LoraConfig] = None,
    ):
        super().__init__(config)
        # AutoModel (no LM head) + last_hidden_state — avoids `output_hidden_states=True`
        # which keeps every layer's hidden in memory and defeats grad checkpointing.
        # torch_dtype="auto" respects the model's config.torch_dtype — for Qwen3
        # this is bf16, which roughly halves base memory vs the fp32 default.
        base = AutoModel.from_pretrained(
            config.base_model_name_or_path,
            torch_dtype='auto',
        )
        hidden_size = base.config.hidden_size
        if lora_config is not None:
            self.llm = get_peft_model(base, lora_config)
        else:
            self.llm = base
        # Scale regression head proportionally to hidden size.
        mid_size = max(256, hidden_size // 4)
        nef = getattr(config, 'n_extra_features', 0) or 0
        self.n_extra_features = nef
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size + nef, mid_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mid_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.max_length = config.max_length
        self.lora_config = lora_config

    def forward(self, input_ids, attention_mask=None, extra_features=None, **kwargs):
        # Returns raw per-sequence predictions (batch, 1). The forecaster and
        # attributer trainers pool these into their own weighted-MSE / KL losses;
        # this module never computes a loss itself.
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)

        if self.config.pooling_method == 'mean':
            mask = (
                attention_mask.unsqueeze(-1)
                if attention_mask is not None
                else torch.ones_like(hidden_states[:, :, :1])
            )
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            # Last non-padded token pooling.
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1).long() - 1
                batch_indices = torch.arange(
                    hidden_states.shape[0], device=hidden_states.device
                )
                pooled = hidden_states[batch_indices, seq_lengths]
            else:
                pooled = hidden_states[:, -1, :]
        # Base LLM may be bf16 while regression_head stays fp32 (HF Trainer
        # autocast handled this during training; inference has no autocast).
        pooled = pooled.to(self.regression_head[0].weight.dtype)
        if getattr(self, 'n_extra_features', 0):
            ef = (
                extra_features.to(pooled.dtype)
                if extra_features is not None
                else pooled.new_zeros(pooled.size(0), self.n_extra_features)
            )
            pooled = torch.cat([pooled, ef], dim=-1)
        out = self.regression_head(pooled)
        if getattr(self.config, 'bounded_output', False):
            out = torch.sigmoid(
                out
            )  # bound per-news pred to [0,1]=target_p; delta=target_p-before respects [-bp,1-bp]
        return out

    # Regressor-specific config fields. These CANNOT live in config.json: for
    # full fine-tuning, self.llm.save_pretrained writes the backbone's own
    # config.json and overwrites anything we put there. We persist them in a
    # sidecar the backbone can't clobber.
    _SIDECAR = 'llm_regressor_config.json'
    _SIDECAR_FIELDS = (
        'base_model_name_or_path',
        'max_length',
        'pooling_method',
        'max_news',
        'target_temperature',
        'predict_delta',
        'bounded_output',
        'n_extra_features',
        'per_news_bce',
    )

    def save_pretrained(self, save_directory: str, state_dict=None, **kwargs):
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        import torch.distributed as dist

        rank0 = (not dist.is_initialized()) or dist.get_rank() == 0

        head_sd_override = None

        # PRIMARY full-FT FSDP path: HF Trainer.save_model gathers the FULL
        # (unsharded) state dict from the FSDP ROOT via accelerator.get_state_dict
        # and passes it here as `state_dict`. The root owns BOTH the backbone
        # (llm.*) AND regression_head.* — so this dict has correct 2-D shapes for
        # everything, including the head. (Our own FSDP.state_dict_type(self,...)
        # could only reach self.llm's child units, leaving the root-owned head as
        # a flat shard.) get_state_dict materializes on rank0; non-rank0 ranks
        # don't enter here, so no collective is needed below.
        if state_dict is not None and self.lora_config is None:
            if not rank0:
                return

            def _clean(k):
                return k.replace('_fsdp_wrapped_module.', '')

            sd = {_clean(k): v for k, v in state_dict.items()}
            backbone_sd = {
                k[len('llm.') :]: v for k, v in sd.items() if k.startswith('llm.')
            }
            head_sd_override = {
                k[len('regression_head.') :]: v
                for k, v in sd.items()
                if k.startswith('regression_head.')
            }
            inner = self.llm.module if hasattr(self.llm, 'module') else self.llm
            inner.save_pretrained(save_directory, state_dict=backbone_sd)
        else:
            # Fallback: FSDP without a passed state_dict (gather self ourselves),
            # or LoRA / single-GPU / DDP (direct save).
            try:
                from torch.distributed.fsdp import FullStateDictConfig
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp import StateDictType

                is_fsdp = any(isinstance(m, FSDP) for m in self.llm.modules())
            except Exception:
                is_fsdp = False
            if is_fsdp and self.lora_config is None:
                cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(self, StateDictType.FULL_STATE_DICT, cfg):
                    full = self.state_dict()  # collective: ALL ranks enter
                if rank0:
                    backbone_sd = {
                        k[len('llm.') :]: v
                        for k, v in full.items()
                        if k.startswith('llm.')
                    }
                    head_sd_override = {
                        k[len('regression_head.') :]: v
                        for k, v in full.items()
                        if k.startswith('regression_head.')
                    }
                    inner = self.llm.module if hasattr(self.llm, 'module') else self.llm
                    inner.save_pretrained(save_directory, state_dict=backbone_sd)
            elif rank0:
                self.llm.save_pretrained(save_directory)

        if not rank0:
            return  # only rank0 writes the head/lora/sidecar files
        if self.lora_config is not None:
            self.lora_config.save_pretrained(save_directory)
        torch.save(
            head_sd_override
            if head_sd_override is not None
            else self.regression_head.state_dict(),
            Path(save_directory) / 'regression_head.bin',
        )
        sidecar = {f: getattr(self.config, f, None) for f in self._SIDECAR_FIELDS}
        with open(Path(save_directory) / self._SIDECAR, 'w') as fh:
            json.dump(sidecar, fh, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        path = Path(pretrained_model_name_or_path)
        sidecar_path = path / cls._SIDECAR
        if sidecar_path.exists():
            with open(sidecar_path) as fh:
                data = json.load(fh)
            config = LLMRegressorConfig(**{k: data.get(k) for k in cls._SIDECAR_FIELDS})
        else:
            # Legacy checkpoints: LoRA kept our fields in config.json; full-FT
            # lost them (backbone config.json won) so they fall back to defaults.
            config = LLMRegressorConfig.from_pretrained(path, **kwargs)

        adapter_present = (path / 'adapter_config.json').exists()
        # Non-LoRA: backbone was saved at this path. Point base loader there.
        # LoRA: backbone wasn't saved; keep config.base_model_name_or_path
        # pointing to the original HF model.
        if not adapter_present:
            config.base_model_name_or_path = str(path)

        model = cls(config, lora_config=None)

        if adapter_present:
            model.llm = PeftModel.from_pretrained(model.llm, str(path))
            model.lora_config = LoraConfig.from_pretrained(str(path))

        head_state = torch.load(
            path / 'regression_head.bin',
            map_location='cpu',
        )
        model.regression_head.load_state_dict(head_state)
        return model
