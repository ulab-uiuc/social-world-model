import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from .data import PolyMarketData
from .utils.filter import TimeBasedPolyMarketFilter
from .utils.utils import unix_to_date


class BaseDataset(Dataset):
    def __init__(self, cache_dir: str = './cache', use_cache: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.use_cache = use_cache
        self.datapoints = []

    def _load_or_create_datapoints(self) -> List[Dict[str, Any]]:
        dataset_hash = self._compute_hash()
        cache_path = self.cache_dir / f'datapoints_{dataset_hash}.pkl'

        if self.use_cache and cache_path.exists():
            try:
                with cache_path.open('rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f'Cache loading failed: {e}')

        datapoints = self._create_datapoints()
        if self.use_cache:
            try:
                with cache_path.open('wb') as f:
                    pickle.dump(datapoints, f)
            except Exception as e:
                print(f'Cache saving failed: {e}')
        return datapoints

    def _compute_hash(self) -> str:
        raise NotImplementedError

    def _create_datapoints(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.datapoints)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.datapoints[idx]


class BasicSocialWMDataset(BaseDataset):
    def __init__(
        self,
        markets: List['PolyMarketData'],
        tokenizer: PreTrainedTokenizer,
        cache_dir: str = './cache',
        window_size: int = 5,
        use_cache: bool = True,
    ):
        super().__init__(cache_dir=cache_dir, use_cache=use_cache)
        self.markets = markets
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.datapoints = self._load_or_create_datapoints()

    def _compute_hash(self) -> str:
        content = [
            f"{m.market_id}{m.start_ts or ''}{m.end_ts or ''}"
            for m in self.markets
            if m.daily_time_series
        ]
        return hashlib.md5(''.join(content).encode()).hexdigest()

    def _create_datapoints(self) -> List[Dict[str, torch.Tensor]]:
        prompts, metadata = [], []

        for market in tqdm(self.markets, desc='Creating datapoints'):
            if not market.daily_time_series or 'Yes' not in market.daily_time_series:
                continue

            series = market.daily_time_series['Yes']
            if len(series) <= self.window_size:
                continue

            for start_idx in range(len(series) - self.window_size):
                window = series[start_idx : start_idx + self.window_size]
                target = series[start_idx + self.window_size]

                prompt = self._build_prompt(market, window, target)
                prompts.append(prompt)
                metadata.append(
                    {
                        'target': target['p'],
                        'market_id': market.market_id,
                        'outcome': 'Yes',
                    }
                )

        encodings = [
            self.tokenizer(p, padding=True, truncation=True, return_tensors='pt')
            for p in tqdm(prompts, desc='Tokenizing')
        ]

        return [
            {
                'input_ids': enc['input_ids'][0],
                'labels': torch.tensor(meta['target'], dtype=torch.float),
                'market_id': meta['market_id'],
                'outcome': meta['outcome'],
            }
            for enc, meta in zip(encodings, metadata)
        ]

    def _build_prompt(self, market, window_data, target_data):
        lines = [f'You are given an event: {market.question}']
        if market.description:
            lines.append(market.description)

        for day in window_data:
            date = unix_to_date(day['t'])
            lines.append(f"At date {date}, its possibility to happen is {day['p']:.3f}")

        target_date = unix_to_date(target_data['t'])
        lines.append(
            f'\nPlease predict the possibility to happen at date {target_date}.'
        )
        return '\n'.join(lines)


class RAGSocialWMDataset(BasicSocialWMDataset):
    def __init__(
        self,
        markets: List['PolyMarketData'],
        similar_markets: Dict[str, List['PolyMarketData']],
        tokenizer: PreTrainedTokenizer,
        cache_dir: str = './cache',
        window_size: int = 5,
        max_sim_markets: int = 3,
        use_cache: bool = True,
    ):
        self.similar_markets = similar_markets
        self.max_sim_markets = max_sim_markets
        super().__init__(
            markets=markets,
            tokenizer=tokenizer,
            cache_dir=cache_dir,
            window_size=window_size,
            use_cache=use_cache,
        )

    def _create_datapoints(self) -> List[Dict[str, torch.Tensor]]:
        prompts, metadata = [], []

        for market in tqdm(self.markets, desc='Creating datapoints'):
            if not market.daily_time_series or 'Yes' not in market.daily_time_series:
                continue

            series = market.daily_time_series['Yes']
            if len(series) <= self.window_size:
                continue

            for start_idx in range(len(series) - self.window_size):
                window = series[start_idx : start_idx + self.window_size]
                target = series[start_idx + self.window_size]

                sim_markets = self.similar_markets.get(market.market_id, [])
                time_overlapped_markets = self._filter_markets(
                    sim_markets, window, target
                )

                prompt = self._build_prompt(
                    market, window, target, time_overlapped_markets
                )

                prompts.append(prompt)
                metadata.append(
                    {
                        'target': target['p'],
                        'market_id': market.market_id,
                        'outcome': 'Yes',
                    }
                )

        encodings = [
            self.tokenizer(p, padding=True, truncation=True, return_tensors='pt')
            for p in tqdm(prompts, desc='Tokenizing')
        ]

        return [
            {
                'input_ids': enc['input_ids'][0],
                'labels': torch.tensor(meta['target'], dtype=torch.float),
                'market_id': meta['market_id'],
                'outcome': meta['outcome'],
            }
            for enc, meta in zip(encodings, metadata)
        ]

    def _filter_markets(self, markets, window_data, target_data):
        filtered_markets = TimeBasedPolyMarketFilter(markets).filter(target_data['t'])
        filtered_markets = filtered_markets[: self.max_sim_markets]

        window_start, window_end = window_data[0]['t'], window_data[-1]['t']

        for market in filtered_markets:
            if 'Yes' not in market.daily_time_series:
                continue
            market.window_series = [
                x
                for x in market.daily_time_series['Yes']
                if window_start <= x['t'] <= window_end
            ]

        return [m for m in filtered_markets if m.window_series]

    def _build_prompt(
        self, market, window_data, target_data, time_overlapped_markets=None
    ):
        lines = super()._build_prompt(market, window_data, target_data).split('\n')

        if time_overlapped_markets:
            lines.append('\nThere are a few similar events:')
            for mkt in time_overlapped_markets:
                lines.append(f'Event: {mkt.question}')
                if mkt.description:
                    lines.append(mkt.description)
                for day in mkt.window_series:
                    date = unix_to_date(day['t'])
                    lines.append(
                        f"At date {date}, its possibility to happen is {day['p']:.3f}"
                    )

        return '\n'.join(lines)
