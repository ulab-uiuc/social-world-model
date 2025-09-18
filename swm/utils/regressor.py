# utils/regressor.py

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from peft import PeftModelForCausalLM

class LLMRegressorConfig(PretrainedConfig):
    model_type = 'llm_regressor'

    def __init__(
        self,
        base_model_name_or_path: Optional[str] = None,
        max_length: Optional[int] = 512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.max_length = max_length


class LLMRegressor(PreTrainedModel):
    config_class = LLMRegressorConfig

    def __init__(
        self,
        config: LLMRegressorConfig,
        lora_config: Optional[LoraConfig] = None,
    ):
        super().__init__(config)

        base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        if lora_config:
            self.llm = get_peft_model(base_model, lora_config)
        else:
            self.llm = base_model

        self.llm.config.pad_token_id = getattr(self.llm.config, "pad_token_id", 0)
        hidden_size = self.llm.config.hidden_size
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.max_length = config.max_length
        self.lora_config = lora_config


    def forward(
        self, input_ids, attention_mask=None, labels=None, weights=None, **kwargs
    ):
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        attention_mask = attention_mask.unsqueeze(-1)
        hidden_states = outputs.hidden_states[-1] * attention_mask
        mean_pooled = hidden_states.sum(dim=1) / attention_mask.sum(dim=1)
        mean_pooled = mean_pooled.float()
        predictions = self.regression_head(mean_pooled)

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

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.llm, "gradient_checkpointing_disable"):
            self.llm.gradient_checkpointing_disable()

    def save_pretrained(self, save_directory: str, **kwargs):
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(save_directory)
        self.lora_config.save_pretrained(save_directory)
        self.llm.save_pretrained(save_directory)

        torch.save(
            self.regression_head.state_dict(),
            Path(save_directory) / 'regression_head.bin',
        )

    # @classmethod
    # def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
    #     config = LLMRegressorConfig.from_pretrained(
    #         pretrained_model_name_or_path, **kwargs
    #     )
    #     try:
    #         lora_config = LoraConfig.from_pretrained(pretrained_model_name_or_path)
    #     except Exception:
    #         lora_config = None

    #     model = cls(config)

    #     model.llm.load_adapter(pretrained_model_name_or_path, adapter_name='default')

    #     model.regression_head.load_state_dict(
    #         torch.load(
    #             Path(pretrained_model_name_or_path) / 'regression_head.bin',
    #             map_location='cpu',
    #         )
    #     )
    #     return model

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        config = LLMRegressorConfig.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        try:
            lora_config = LoraConfig.from_pretrained(pretrained_model_name_or_path)
        except Exception:
            lora_config = None

        model = cls(config, lora_config=lora_config)

        if lora_config:
            model.llm.load_adapter(pretrained_model_name_or_path, adapter_name='default')

        model.regression_head.load_state_dict(
            torch.load(
                Path(pretrained_model_name_or_path) / 'regression_head.bin',
                map_location='cpu',
            )
        )
        return model

