from typing import Dict, List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .data import PolyMarketData
from .utils.utils import unix_to_date


class PolyMarketDataset(Dataset):
    def __init__(
        self,
        markets: List[PolyMarketData],
        similar_markets_dict: Dict[str, List[PolyMarketData]],
        tokenizer,
        window_size: int = 5,
        max_series_length: int = 50,
    ):
        self.tokenizer = tokenizer
        self.datapoints = []
        self._create_datapoints(
            markets, similar_markets_dict, window_size, max_series_length
        )

    def _create_datapoints(
        self, markets, similar_markets_dict, window_size, max_series_length
    ):
        for market in tqdm(markets):
            if not market.time_series:
                continue

            similar_markets = similar_markets_dict[market.market_id]

            for outcome, series in market.time_series.items():
                if len(series) > max_series_length:
                    gap = len(series) // max_series_length
                    series = series[::gap]
                if len(series) <= window_size:
                    continue

                for i in range(len(series) - window_size):
                    window = series[i : i + window_size]
                    target = series[i + window_size]

                    prompt = f'You are given an event: {market.question}\n'
                    if market.description:
                        prompt += f'{market.description}\n'

                    for p in window:
                        prompt += f"At date {unix_to_date(p['t'])}, its winning possibility is {p['p']:.3f}\n"

                    prompt += '\nThere are a few similar events:\n'
                    for m in similar_markets:
                        if m.market_id != market.market_id:
                            prompt += f'Event: {m.question}\n'
                            if len(m.time_series.get(outcome, [])) > 0:
                                latest = m.time_series[outcome][-1]
                                prompt += f"At date {unix_to_date(latest['t'])}, its winning possibility is {latest['p']:.3f}\n"

                    target_date = unix_to_date(target['t'])
                    prompt += f'\nPlease predict the winning possibility at date {target_date}.'

                    encodings = self.tokenizer(
                        prompt, padding=True, truncation=True, return_tensors='pt'
                    )
                    self.datapoints.append(
                        {
                            'input_ids': encodings['input_ids'][0],
                            'attention_mask': encodings['attention_mask'][0],
                            'labels': torch.tensor(target['p'], dtype=torch.float),
                        }
                    )

        print(f'Created {len(self.datapoints)} datapoints')

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        return self.datapoints[idx]
