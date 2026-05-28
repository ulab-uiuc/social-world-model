# utils/regressor.py

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
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

    def __init__(self, config: LLMRegressorConfig):
        super().__init__(config)
        self.llm = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        hidden_size = self.llm.config.hidden_size
        # Scale regression head proportionally to hidden size.
        mid_size = max(256, hidden_size // 4)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, mid_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mid_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.max_length = config.max_length

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
            mask = attention_mask.unsqueeze(-1) if attention_mask is not None else torch.ones_like(hidden_states[:, :, :1])
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            # Last non-padded token pooling.
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1).long() - 1
                batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
                pooled = hidden_states[batch_indices, seq_lengths]
            else:
                pooled = hidden_states[:, -1, :]
        predictions = self.regression_head(pooled)

        if labels is not None:
            if weights is not None:
                loss = torch.mean(weights * (predictions.view(-1) - labels.view(-1)) ** 2)
            else:
                loss = nn.MSELoss()(predictions.view(-1), labels.view(-1))
            return {'loss': loss, 'predictions': predictions}
        return predictions

    def save_pretrained(self, save_directory: str, **kwargs):
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(save_directory)
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
        config.base_model_name_or_path = pretrained_model_name_or_path
        model = cls(config)

        head_state = torch.load(
            Path(pretrained_model_name_or_path) / 'regression_head.bin',
            map_location='cpu',
        )
        model.regression_head.load_state_dict(head_state)
        return model
