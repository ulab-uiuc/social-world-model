# utils/regressor.py

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel


class LLMRegressorConfig(PretrainedConfig):
    model_type = 'llm_regressor'

    def __init__(
        self,
        base_model_name_or_path: Optional[str] = None,
        max_length: Optional[int] = 512,
        pooling_method: Optional[str] = 'last_token',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.max_length = max_length
        self.pooling_method = pooling_method


class LLMRegressor(PreTrainedModel):
    config_class = LLMRegressorConfig

    def __init__(
        self,
        config: LLMRegressorConfig,
        lora_config: Optional[LoraConfig] = None,
    ):
        super().__init__(config)
        self.llm = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        hidden_size = self.llm.config.hidden_size
        # Scale regression head proportionally to hidden size
        # 0.6B: hidden=1024 → 256 (4x compression)
        # 8B:   hidden=4096 → 1024 → 256 (4x each step)
        mid_size = max(256, hidden_size // 4)  # Proportional first layer
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, mid_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mid_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.max_length = config.max_length
        self.lora_config = lora_config
        if lora_config:
            self.llm = get_peft_model(self.llm, lora_config)

    def forward(
        self, input_ids, attention_mask=None, labels=None, weights=None, **kwargs
    ):
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden)

        if self.config.pooling_method == 'mean':
            # Mean pooling over non-padded tokens (original method)
            mask = attention_mask.unsqueeze(-1) if attention_mask is not None else torch.ones_like(hidden_states[:, :, :1])
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            # Last non-padded token pooling (better for causal decoder models)
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1).long() - 1
                batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
                pooled = hidden_states[batch_indices, seq_lengths]
            else:
                pooled = hidden_states[:, -1, :]
        predictions = self.regression_head(pooled)

        loss = None
        if labels is not None:
            if weights is not None:
                loss = torch.mean(
                    weights * (predictions.view(-1) - labels.view(-1)) ** 2
                )
            else:
                loss = nn.MSELoss()(predictions.view(-1), labels.view(-1))
            return {'loss': loss, 'predictions': predictions}
        return predictions

    def save_pretrained(self, save_directory: str, **kwargs):
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(save_directory)
        if self.lora_config is not None:
            self.lora_config.save_pretrained(save_directory)
        self.llm.save_pretrained(save_directory)

        torch.save(
            self.regression_head.state_dict(),
            Path(save_directory) / 'regression_head.bin',
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        config = LLMRegressorConfig.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        adapter_config_path = (
            Path(pretrained_model_name_or_path) / 'adapter_config.json'
        )

        # Load regression head first to detect old vs new format
        head_state = torch.load(
            Path(pretrained_model_name_or_path) / 'regression_head.bin',
            map_location='cpu',
        )
        # Detect checkpoint format by inspecting head weights:
        # - LayerNorm format: '0.weight' is 1D (LayerNorm params)
        # - No-LayerNorm format: '0.weight' is 2D (Linear params)
        has_layernorm = '0.weight' in head_state and head_state['0.weight'].dim() == 1

        # Old checkpoints don't have pooling_method saved in config.json.
        import json
        config_json_path = Path(pretrained_model_name_or_path) / 'config.json'
        if config_json_path.exists():
            with open(config_json_path) as f:
                raw_config = json.load(f)
            if 'pooling_method' not in raw_config:
                # Very old checkpoints (no LN, no pooling_method) used mean pooling
                config.pooling_method = 'mean' if not has_layernorm else 'last_token'
        print(f"Loaded checkpoint with pooling_method={config.pooling_method}, has_layernorm={has_layernorm}")

        if adapter_config_path.exists():
            lora_config = LoraConfig.from_pretrained(pretrained_model_name_or_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path
            )
            model = cls(config, lora_config=lora_config)
            model.llm = get_peft_model(base_model, lora_config)
            model.llm.load_adapter(
                pretrained_model_name_or_path, adapter_name='default'
            )
        else:
            # Non-LoRA checkpoint: load backbone directly from checkpoint directory.
            config.base_model_name_or_path = pretrained_model_name_or_path
            model = cls(config, lora_config=None)

        if has_layernorm:
            # Checkpoint saved with LayerNorm — rebuild head with LayerNorm
            hidden_size = head_state['0.weight'].shape[0]  # LayerNorm size
            model.regression_head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        model.regression_head.load_state_dict(head_state)
        return model
