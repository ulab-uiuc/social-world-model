from typing import Dict, List

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data import PolyMarketData


class LLMRegression(nn.Module):
    def __init__(
        self, model_name: str = 'mistralai/Mistral-7B-v0.1', max_length: int = 512
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        self.regression_head = nn.Sequential(
            nn.Linear(self.llm.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.max_length = max_length

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # Last token of last layer
        predictions = self.regression_head(last_hidden)

        loss = None
        if labels is not None:
            loss = nn.MSELoss()(predictions, labels.unsqueeze(-1))

        return (
            {'loss': loss, 'predictions': predictions}
            if loss is not None
            else predictions
        )

    def predict(
        self, market: PolyMarketData, similar_markets: List[PolyMarketData]
    ) -> Dict[str, float]:
        self.eval()
        with torch.no_grad():
            similar_contexts = [
                f"Similar market question: {m.question}\nDescription: {m.discrption or ''}"
                for m in similar_markets
                if m.market_id != market.market_id
            ]

            predictions = {}
            for outcome, series in market.time_series.items():
                if not series:
                    continue

                context = f'Question: {market.question}\n'
                if market.discrption:
                    context += f'Description: {market.discrption}\n'
                context += '\n'.join(similar_contexts) + '\n'

                window = series[-self.window_size :]
                series_text = ' '.join([f"{p['value']:.3f}" for p in window])
                prompt = f'{context}Recent values: {series_text}\nPredict next value for {outcome}:'

                inputs = self.tokenizer(
                    prompt,
                    return_tensors='pt',
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                )
                pred = self(**inputs)
                predictions[outcome] = torch.sigmoid(pred).item()

        return predictions
