from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMRegressor(nn.Module):
    def __init__(self, model_name: str, max_length: int = 1024):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        self.regression_head = nn.Sequential(
            nn.Linear(self.llm.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(self.device)

        self.max_length = max_length

    def to(self, device):
        self.device = device
        self.llm = self.llm.to(device)
        self.regression_head = self.regression_head.to(device)
        return self

    def save_pretrained(self, path: str, safe_serialization: bool = True):
        path = Path(path)
        path.mkdir(exist_ok=True)
        self.llm.save_pretrained(path / 'llm', safe_serialization=False)
        torch.save(self.regression_head.state_dict(), path / 'regression_head.pt')

    @classmethod
    def from_pretrained(cls, path: str):
        path = Path(path)
        model = cls()
        model.llm = AutoModelForCausalLM.from_pretrained(path / 'llm')
        regression_head_state = torch.load(path / 'regression_head.pt')
        model.regression_head.load_state_dict(regression_head_state)
        return model

    def forward(self, input_ids, attention_mask=None, labels=None):
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        last_hidden = outputs.hidden_states[-1][:, -1, :]
        predictions = self.regression_head(last_hidden)

        loss = None
        if labels is not None:
            loss = nn.MSELoss()(predictions, labels.unsqueeze(-1))

        return (
            {'loss': loss, 'predictions': predictions}
            if loss is not None
            else predictions
        )
