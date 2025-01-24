# utils/regressor.py

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path
from typing import Optional
from peft import LoraConfig, get_peft_model


class LLMRegressorConfig(PretrainedConfig):
    model_type = "llm_regressor"
    
    def __init__(self, base_model_name_or_path="gpt2", max_length=1024, **kwargs):
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
        self.llm = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
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
        if lora_config:
            self.llm = get_peft_model(self.llm, lora_config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        predictions = self.regression_head(last_hidden)
        
        loss = None
        if labels is not None:
            loss = nn.MSELoss()(predictions.view(-1), labels.view(-1))
            return {"loss": loss, "predictions": predictions}
        
        return predictions

    def save_pretrained(self, save_directory: str, **kwargs):
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        self.llm.save_pretrained(save_directory, **kwargs)
        torch.save(self.regression_head.state_dict(), Path(save_directory) / 'regression_head.bin')
        tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path: str, 
        *model_args, 
        **kwargs
    ):
        config = LLMRegressorConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        lora_config_path = Path(pretrained_model_name_or_path) / 'peft_config.json'
        lora_config = None
        if lora_config_path.exists():
            with open(lora_config_path, 'r') as f:
                lora_config_dict = json.load(f)
                lora_config = LoraConfig.from_dict(lora_config_dict)
        model = cls(config, lora_config=lora_config)
        regression_head_path = Path(pretrained_model_name_or_path) / 'regression_head.bin'
        if regression_head_path.exists():
            model.regression_head.load_state_dict(torch.load(regression_head_path, map_location='cpu'))
        return model
