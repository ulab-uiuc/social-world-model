from typing import Dict, List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .data import PolyMarketData
from .utils.utils import unix_to_date
from pprint import pprint

class PolyMarketDataset(Dataset):
    def __init__(
        self,
        markets: List[PolyMarketData],
        similar_markets_dict: Dict[str, List[PolyMarketData]],
        tokenizer,
        window_size: int = 20,
        max_series_length: int = 50,
        max_sim_markets: int = 10,
    ):
        self.tokenizer = tokenizer
        self.datapoints = []
        self._create_datapoints(
            markets, similar_markets_dict, window_size, max_series_length, max_sim_markets
        )

    def _filter_midnight_points(self, series):
        from datetime import datetime
        midnight_points = []
        for point in series:
            dt = datetime.fromtimestamp(point['t'])
            if dt.hour == 0 and dt.minute == 0:
                midnight_points.append(point)
        return midnight_points

    def _create_datapoints(
        self, markets, similar_markets_dict, window_size, max_series_length, max_sim_markets
    ):
        for market in tqdm(markets):
            if not market.time_series:
                continue
                
            for outcome, series in market.time_series.items():
                daily_series = self._filter_midnight_points(series)
                if len(daily_series) <= window_size:
                    continue
                    
                for i in range(len(daily_series) - window_size):
                    window = daily_series[i : i + window_size]
                    target = daily_series[i + window_size]
                    window_start_time = window[0]['t']
                    window_end_time = window[-1]['t']
                    target_time = target['t']

                    relevant_similar_markets = []
                    similar_markets = similar_markets_dict[market.market_id]
                    
                    for sim_market in similar_markets:
                        if sim_market.market_id == market.market_id or outcome not in sim_market.time_series:
                            continue
                            
                        sim_series = self._filter_midnight_points(sim_market.time_series[outcome])
                        window_series = []
                        
                        for entry in sim_series:
                            if window_start_time <= entry['t'] <= window_end_time:
                                window_series.append(entry)
                                
                        if window_series:
                            sim_market.window_series = window_series
                            relevant_similar_markets.append(sim_market)

                    prompt = self._create_prompt(
                        market, window, target, relevant_similar_markets, max_sim_markets
                    )
                    pprint(prompt)
                    import pdb; pdb.set_trace()
                    encodings = self.tokenizer(
                        prompt, padding=True, truncation=True, return_tensors='pt'
                    )
                    
                    self.datapoints.append({
                        'input_ids': encodings['input_ids'][0],
                        'attention_mask': encodings['attention_mask'][0],
                        'labels': torch.tensor(target['p'], dtype=torch.float),
                        'market_id': market.market_id,
                        'outcome': outcome
                    })

    def _create_prompt(self, market, window, target, similar_markets, max_sim_markets):
        prompt = f'You are given an event: {market.question}\n'
        if market.description:
            prompt += f'{market.description}\n'
            
        for p in window:
            prompt += f"At date {unix_to_date(p['t'])}, its winning possibility is {p['p']:.3f}\n"
            
        if similar_markets:
            prompt += '\nThere are a few similar events:\n'
            for m in similar_markets[:max_sim_markets]:
                prompt += f'Event: {m.question}\n'
                if m.description:
                    prompt += f'{m.description}\n'
                for point in m.window_series:
                    prompt += f"At date {unix_to_date(point['t'])}, its winning possibility is {point['p']:.3f}\n"

        target_date = unix_to_date(target['t'])
        prompt += f'\nPlease predict the winning possibility at date {target_date}.'
        
        return prompt

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        return self.datapoints[idx]