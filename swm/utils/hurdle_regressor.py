"""Hurdle-formulation regressor for zero-inflated continuous targets like
prediction-market price deltas.

Two heads on the same pooled hidden state from a (LoRA-adapted) LLM:
  - gate_head : P(|Δ| >= gate_threshold | x)             — BCE loss
  - mag_head  : HL-Gauss distribution over Δ | Δ ≠ 0    — soft-label CE

Inference:  pred_Δ = sigmoid(gate_logit) * E_HL-Gauss[mag]

This matches the structural fact that targets are a delta-at-0 + continuous mix.
Standard MSE / Gaussian-NLL heads work under wrong likelihood assumption and
get pulled to predicting the mean (≈ 0); the hurdle factorisation gives both
heads clean gradient signal that doesn't pull either toward 0.

Bin design (defaults):
  - 41 bins evenly spaced in [-0.5, +0.5] (covers has-news IQR ±0.11 + tails)
  - HL-Gauss sigma = 0.04 (≈1.6 bin widths, smooth but not too soft)
  - gate_threshold = 0.02 (matches "non-flat" definition used in diag)
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel


class HurdleLLMRegressorConfig(PretrainedConfig):
    model_type = 'hurdle_llm_regressor'

    def __init__(
        self,
        base_model_name_or_path: str = '',
        max_length: int = 1024,
        pooling_method: str = 'last_token',
        n_mag_bins: int = 41,
        mag_range: float = 0.5,
        gauss_sigma: float = 0.04,
        gate_threshold: float = 0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.max_length = max_length
        self.pooling_method = pooling_method
        self.n_mag_bins = n_mag_bins
        self.mag_range = mag_range
        self.gauss_sigma = gauss_sigma
        self.gate_threshold = gate_threshold


class HurdleLLMRegressor(PreTrainedModel):
    config_class = HurdleLLMRegressorConfig

    def __init__(
        self,
        config: HurdleLLMRegressorConfig,
        lora_config: Optional[LoraConfig] = None,
    ):
        super().__init__(config)
        self.llm = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        hidden_size = self.llm.config.hidden_size
        mid_size = max(256, hidden_size // 4)
        # Gate head: scalar logit for P(active)
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_size, mid_size),
            nn.ReLU(),
            nn.Linear(mid_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Magnitude head: logits over n_mag_bins (categorical → HL-Gauss target)
        self.mag_head = nn.Sequential(
            nn.Linear(hidden_size, mid_size),
            nn.ReLU(),
            nn.Linear(mid_size, 128),
            nn.ReLU(),
            nn.Linear(128, config.n_mag_bins),
        )
        # Bin centres (registered so they move with .to(device))
        centres = torch.linspace(-config.mag_range, config.mag_range, config.n_mag_bins)
        self.register_buffer('bin_centres', centres)
        self.lora_config = lora_config
        if lora_config is not None:
            self.llm = get_peft_model(self.llm, lora_config)

    # ---------------------------- forward ----------------------------------

    def _pool(self, hidden_states, attention_mask):
        if self.config.pooling_method == 'mean':
            if attention_mask is None:
                return hidden_states.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-8)
        # last non-pad token
        if attention_mask is None:
            return hidden_states[:, -1, :]
        seq_lens = attention_mask.sum(dim=1).long() - 1
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, seq_lens]

    def forward(self, input_ids, attention_mask=None, **kwargs):
        out = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        h = out.hidden_states[-1]
        pooled = self._pool(h, attention_mask)
        gate_logits = self.gate_head(pooled).squeeze(-1)  # (B,)
        mag_logits = self.mag_head(pooled)  # (B, n_bins)
        return {'gate_logits': gate_logits, 'mag_logits': mag_logits, 'pooled': pooled}

    # ----------------------- hurdle math helpers ---------------------------

    def hl_gauss_target(self, target):
        """target: (B,) -> smoothed prob distribution (B, n_bins)."""
        diff = (
            target.unsqueeze(-1) - self.bin_centres.unsqueeze(0)
        ) / self.config.gauss_sigma
        log_p = -0.5 * diff * diff
        return torch.softmax(log_p, dim=-1)

    def predict_delta(self, gate_logits, mag_logits):
        """Hurdle prediction: P(active) * E[mag | active]."""
        p_active = torch.sigmoid(gate_logits)
        mag_probs = torch.softmax(mag_logits, dim=-1)
        mag_exp = (mag_probs * self.bin_centres.unsqueeze(0)).sum(dim=-1)
        return p_active * mag_exp

    # ----------------------- save / load -----------------------------------

    def save_pretrained(self, save_directory: str, **kwargs):
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(save_directory)
        if self.lora_config is not None:
            self.lora_config.save_pretrained(save_directory)
        self.llm.save_pretrained(save_directory)
        torch.save(
            {
                'gate': self.gate_head.state_dict(),
                'mag': self.mag_head.state_dict(),
                'bin_centres': self.bin_centres.detach().cpu(),
            },
            Path(save_directory) / 'hurdle_heads.bin',
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        path = Path(pretrained_model_name_or_path)
        config = HurdleLLMRegressorConfig.from_pretrained(path, **kwargs)
        adapter_path = path / 'adapter_config.json'

        # Build skeleton without LoRA, then attach pretrained LoRA if present
        model = cls(config, lora_config=None)
        if adapter_path.exists():
            model.llm = PeftModel.from_pretrained(model.llm, str(path))
            # Load LoRA config back for save round-trip
            model.lora_config = LoraConfig.from_pretrained(str(path))

        heads = torch.load(
            path / 'hurdle_heads.bin', map_location='cpu', weights_only=False
        )
        model.gate_head.load_state_dict(heads['gate'])
        model.mag_head.load_state_dict(heads['mag'])
        if 'bin_centres' in heads:
            model.bin_centres = heads['bin_centres']
        return model
