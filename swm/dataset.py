import hashlib
import pickle
from pathlib import Path
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
        cache_dir: str = './cache',
        window_size: int = 5,
        max_sim_markets: int = 3,
        use_cache: bool = True,
    ):
        self.markets = markets
        self.similar_markets_dict = similar_markets_dict
        self.tokenizer = tokenizer
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.window_size = window_size
        self.max_sim_markets = max_sim_markets
        self.use_cache = use_cache

        self.datapoints = self._prepare_datapoints()

    def __len__(self) -> int:
        return len(self.datapoints)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.datapoints[idx]

    def _prepare_datapoints(self) -> List[Dict[str, torch.Tensor]]:
        dataset_hash = self._compute_dataset_hash(self.markets, self.window_size)
        cache_path = self._get_cache_path(dataset_hash)

        if self.use_cache and cache_path.exists():
            try:
                with cache_path.open('rb') as f:
                    print(f'Loading dataset from cache: {cache_path}')
                    return pickle.load(f)
            except Exception as e:
                print(f'Cache loading failed ({cache_path}): {e}')

        datapoints = self._create_datapoints()
        if self.use_cache:
            try:
                with cache_path.open('wb') as f:
                    print(f'Saving dataset to cache: {cache_path}')
                    pickle.dump(datapoints, f)
            except Exception as e:
                print(f'Cache saving failed ({cache_path}): {e}')

        return datapoints

    def _compute_dataset_hash(
        self, markets: List[PolyMarketData], window_size: int
    ) -> str:
        hash_content = []
        for market in markets:
            if market.daily_time_series:
                hash_content.extend(
                    [
                        market.market_id,
                        str(market.start_ts or ''),
                        str(market.end_ts or ''),
                        str(window_size),
                    ]
                )
        return hashlib.md5(''.join(hash_content).encode()).hexdigest()

    def _get_cache_path(self, dataset_hash: str) -> Path:
        return self.cache_dir / f'datapoints_{dataset_hash}.pkl'

    def _create_datapoints(self) -> List[Dict[str, torch.Tensor]]:
        prompts = []
        metadata = []

        for market in tqdm(self.markets, desc='Creating datapoints'):
            if not market.daily_time_series:
                continue

            if 'Yes' not in market.daily_time_series:
                continue

            series = market.daily_time_series['Yes']

            if len(series) <= self.window_size:
                continue

            sim_markets = [
                m
                for m in self.similar_markets_dict.get(market.market_id, [])
                if m.market_id != market.market_id and m.start_ts and m.end_ts
            ]

            for start_idx in range(len(series) - self.window_size):
                window_data = series[start_idx : start_idx + self.window_size]
                target_data = series[start_idx + self.window_size]

                relevant_markets = self._filter_similar_markets(
                    sim_markets, 'Yes', window_data, target_data
                )
                if not relevant_markets:
                    continue

                prompt_text = self._build_prompt(
                    market, window_data, target_data, relevant_markets
                )
                prompts.append(prompt_text)
                metadata.append(
                    {
                        'target': target_data['p'],
                        'market_id': market.market_id,
                        'outcome': 'Yes',
                    }
                )

        encodings = [
            self.tokenizer(prompt, padding=True, truncation=True, return_tensors='pt')
            for prompt in tqdm(prompts, desc='Tokenizing prompts')
        ]

        datapoints = []
        for i, meta in enumerate(metadata):
            datapoints.append(
                {
                    'input_ids': encodings[i]['input_ids'][0],
                    'labels': torch.tensor(meta['target'], dtype=torch.float),
                    'market_id': meta['market_id'],
                    'outcome': meta['outcome'],
                }
            )

        return datapoints

    def _filter_similar_markets(
        self,
        similar_markets: List[PolyMarketData],
        outcome: str,
        window_data: List[Dict[str, float]],
        target_data: Dict[str, float],
    ) -> List[PolyMarketData]:
        window_start_ts = window_data[0]['t']
        window_end_ts = window_data[-1]['t']
        target_ts = target_data['t']

        filtered = []
        for sim_mkt in similar_markets:
            if not (sim_mkt.start_ts <= target_ts <= sim_mkt.end_ts):
                continue

            if outcome not in sim_mkt.daily_time_series:
                continue

            sim_mkt.window_series = [
                x
                for x in sim_mkt.daily_time_series[outcome]
                if window_start_ts <= x['t'] <= window_end_ts
            ]
            filtered.append(sim_mkt)

        return filtered[: self.max_sim_markets]

    def _build_prompt(
        self,
        market: PolyMarketData,
        window_data: List[Dict[str, float]],
        target_data: Dict[str, float],
        relevant_markets: List[PolyMarketData],
    ) -> str:
        prompt_lines = [
            f'You are given an event: {market.question}',
        ]
        if market.description:
            prompt_lines.append(market.description)

        for day_data in window_data:
            date_str = unix_to_date(day_data['t'])
            prompt_lines.append(
                f"At date {date_str}, its possibility to happen is {day_data['p']:.3f}"
            )

        target_date_str = unix_to_date(target_data['t'])
        prompt_lines.append(
            f'\nPlease predict the possibility to happen at date {target_date_str}.'
        )

        if relevant_markets:
            prompt_lines.append('\nThere are a few similar events:')
            for sim_mkt in relevant_markets:
                prompt_lines.append(f'Event: {sim_mkt.question}')
                if sim_mkt.description:
                    prompt_lines.append(sim_mkt.description)
                for sim_day_data in sim_mkt.window_series:
                    sim_date_str = unix_to_date(sim_day_data['t'])
                    prompt_lines.append(
                        f"At date {sim_date_str}, its possibility to happen is {sim_day_data['p']:.3f}"
                    )

        return '\n'.join(prompt_lines)
