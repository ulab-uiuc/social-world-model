from typing import Dict, List

import torch
from datasets import Dataset

from .data import PolyMarketData


class MarketDataset(Dataset):
    def __init__(
        self,
        markets: List[PolyMarketData],
        similar_markets_dict: Dict[str, List[PolyMarketData]],
        tokenizer,
        window_size: int = 5,
    ):
        self.datapoints = []
        self.tokenizer = tokenizer
        self.window_size = window_size
        self._create_datapoints(markets, similar_markets_dict)

    def _create_datapoints(self, markets, similar_markets_dict):
        for market in markets:
            if not market.time_series:
                continue

            similar_markets = similar_markets_dict[market.market_id]
            similar_contexts = [
                f"Similar market question: {m.question}\nDescription: {m.discrption or ''}"
                for m in similar_markets
                if m.market_id != market.market_id
            ]

            for outcome, series in market.time_series.items():
                if len(series) <= self.window_size:
                    continue

                for i in range(len(series) - self.window_size):
                    window = series[i : i + self.window_size]
                    target = series[i + self.window_size]['value']

                    context = f'Question: {market.question}\n'
                    if market.discrption:
                        context += f'Description: {market.discrption}\n'
                    context += '\n'.join(similar_contexts) + '\n'
                    series_text = ' '.join([f"{p['value']:.3f}" for p in window])
                    prompt = f'{context}Recent values: {series_text}\nPredict next value for {outcome}:'

                    encodings = self.tokenizer(
                        prompt, padding=True, truncation=True, return_tensors='pt'
                    )
                    self.datapoints.append(
                        {
                            'input_ids': encodings['input_ids'][0],
                            'attention_mask': encodings['attention_mask'][0],
                            'labels': target,
                            'market_id': market.market_id,
                            'outcome': outcome,
                        }
                    )

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        point = self.datapoints[idx]
        return {
            'input_ids': point['input_ids'],
            'attention_mask': point['attention_mask'],
            'labels': torch.tensor(point['labels'], dtype=torch.float),
        }
