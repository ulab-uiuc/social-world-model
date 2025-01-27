import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from collections import defaultdict

from .data import PolyMarketData
from .utils.filter import TimeBasedPolyMarketFilter
from .utils.utils import unix_to_date
from .reasoner import BasicPosteriorReasoner, BasicPriorReasoner


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


class BasicPolyMarketDataset(BaseDataset):
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
                        'market_id': market.market_id,
                        'event_id': market.event_id,
                        'label': target['p'],
                        't': target['t'],
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
                'label': torch.tensor(meta['label'], dtype=torch.float),
                'market_id': meta['market_id'],
                'event_id': meta['event_id'],
                't': meta['t'],
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


class RAGPolyMarketDataset(BasicPolyMarketDataset):
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
                        'market_id': market.market_id,
                        'event_id': market.event_id,
                        'label': target['p'],
                        't': target['t'],
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
                'label': torch.tensor(meta['label'], dtype=torch.float),
                'market_id': meta['market_id'],
                'event_id': meta['event_id'],
                't': meta['t'],
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


class BasicPolyMarketDatasetWithEvent(BaseDataset):
    def __init__(
        self,
        markets: List['PolyMarketData'],
        tokenizer: PreTrainedTokenizer,
        reasoner: Union[BasicPosteriorReasoner, BasicPriorReasoner],
        cache_dir: str = './cache',
        window_size: int = 5,
        use_cache: bool = True,
    ):
        super().__init__(cache_dir=cache_dir, use_cache=use_cache)
        self.reasoner = reasoner
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
        content_hash = hashlib.md5(''.join(content).encode()).hexdigest()
        reasoner_hash = hashlib.md5(
            str(self.reasoner.__dict__['model_name']).encode()
        ).hexdigest()
        return hashlib.md5(f'{content_hash}{reasoner_hash}'.encode()).hexdigest()

    def _create_datapoints(self) -> List[Dict[str, Any]]:
        grouped_points = defaultdict(
            lambda: {
                'input_ids': [],
                'attention_mask': [],
                'weights': [],
                'label': None,   # we will set the label eventually
                'market_id': None,
                't': None
            }
        )
        all_prompts = []
        prompt_to_meta = {}

        # 1) Build all prompts
        for market in tqdm(self.markets, desc='Creating prompts'):
            if not market.daily_time_series or 'Yes' not in market.daily_time_series:
                continue

            series = market.daily_time_series['Yes']
            if len(series) <= self.window_size:
                continue

            for start_idx in range(len(series) - self.window_size):
                window = series[start_idx : start_idx + self.window_size]
                target = series[start_idx + self.window_size]
                current_ts = window[-1]['t']
                
                events = self.reasoner.reason(current_ts, market)
                for event in events:
                    prompt = self._build_prompt(market, window, target, event['news'])
                    all_prompts.append(prompt)
                    
                    key = (market.market_id, target['t'])
                    prompt_to_meta[len(all_prompts) - 1] = {
                        'key': key,
                        'score': event['score'],
                        'label': target['p'],
                        'market_id': market.market_id,
                        't': target['t']
                    }

        # 2) Batch tokenization (collect both input_ids and attention_mask)
        batch_size = 1000
        for i in tqdm(range(0, len(all_prompts), batch_size), desc='Tokenizing'):
            batch_prompts = all_prompts[i : i + batch_size]
            
            encodings = self.tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                return_attention_mask=True,
            )
            
            # encodings['input_ids'].shape = [batch_size, seq_len]
            # encodings['attention_mask'].shape = [batch_size, seq_len]

            for j in range(encodings['input_ids'].size(0)):
                global_idx = i + j  # index in the entire list of prompts
                meta = prompt_to_meta[global_idx]
                key = meta['key']

                # Initialize if not present
                if grouped_points[key]['label'] is None:
                    grouped_points[key]['label'] = torch.tensor(
                        meta['label'], dtype=torch.float
                    )
                    grouped_points[key]['market_id'] = meta['market_id']
                    grouped_points[key]['t'] = meta['t']

                grouped_points[key]['input_ids'].append(encodings['input_ids'][j])
                grouped_points[key]['attention_mask'].append(encodings['attention_mask'][j])
                grouped_points[key]['weights'].append(meta['score'])

        # 3) Convert weights to tensor and stack/pad the input_ids/attention_mask
        final_datapoints = []
        for key, group in grouped_points.items():
            # (a) Convert weights to tensor
            weights_tensor = torch.tensor(group['weights'], dtype=torch.float)

            # (b) Stack input_ids and attention_masks
            # If you use 'pad_sequence', note that each prompt can have different seq lengths
            # Here we rely on the *same* truncation/padding from the single batch tokenizer step,
            # so they should all have the same length if we used `padding=True` + `truncation=True`.
            # If for some reason they differ in length, you can use pad_sequence on each list:
            #
            #   input_ids_tensor = pad_sequence(group['input_ids'], batch_first=True, padding_value=self.tokenizer.pad_token_id)
            #   attn_mask_tensor = pad_sequence(group['attention_mask'], batch_first=True, padding_value=0)
            #
            # However, if the above single-batch tokenization is consistent, 
            # you can simply do torch.stack:
            input_ids_tensor = torch.stack(group['input_ids'], dim=0)
            attn_mask_tensor = torch.stack(group['attention_mask'], dim=0)

            final_datapoints.append({
                'input_ids': input_ids_tensor,  # shape [num_events, seq_len]
                'attention_mask': attn_mask_tensor,  # shape [num_events, seq_len]
                'label': group['label'],  # shape [], we can repeat or expand in __getitem__
                'weights': weights_tensor,  # shape [num_events]
                'market_id': group['market_id'],
                't': group['t']
            })

        return final_datapoints

    def _build_prompt(self, market: 'PolyMarketData', window_data: List[Dict], target_data: Dict, news: Dict) -> str:
        lines = [f'You are given an event: {market.question}']
        if market.description:
            lines.append(market.description)

        for day in window_data:
            date = unix_to_date(day['t'])
            lines.append(f"At date {date}, its possibility to be 'Yes' to the event is {day['p']:.3f}")

        target_date = unix_to_date(target_data['t'])
        news_content = (
            f"The date of the news is {news.date}.\n"
            f"The title is {news.title}.\n"
            f"{news.description}"
        )
        lines.append(
            f'\nPlease predict the possibility to happen at date {target_date} based on the following news:\n{news_content}'
        )
        return '\n'.join(lines)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.datapoints[idx]

        num_events = item['input_ids'].size(0)
        label_tensor = item['label'].expand(num_events)

        return {
            'input_ids': item['input_ids'],           # (num_events, seq_len)
            'attention_mask': item['attention_mask'], # (num_events, seq_len)
            'label': label_tensor,                    # (num_events,)
            'weights': item['weights'],               # (num_events,)
            'market_key': torch.tensor(
                [hash(item['market_id']), item['t']], dtype=torch.long
            ),
            'market_id': item['market_id'],
            'event_id': item['market_id'],
            't': item['t'],
        }