"""
Simplified Dataset classes that directly consume preprocessed data.

The key insight: all complex logic (windowing, attribution, news matching)
should happen in the preprocessing/converter stage. Datasets just load and format.
"""
import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from .data import MarketData
from .utils.utils import unix_to_date


class BaseDataset(Dataset):
    """Base dataset with optional caching."""
    def __init__(self, cache_dir: str, use_cache: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
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
        print(f'Created {len(datapoints)} datapoints')
        return datapoints

    def _compute_hash(self) -> str:
        raise NotImplementedError

    def _create_datapoints(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.datapoints)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.datapoints[idx]


class MultiEventForecasterDataset(BaseDataset):
    """
    Simple dataset for MultiEventForecaster.
    
    Expects each market's daily_breakpoints to already contain:
    - window_history: List of {t, p} for the input window
    - after: {t, p} for the target
    - news: List of news items
    - attributions: List of {news_idx, score} for weighting
    
    Dataset just formats this into training examples.
    """
    def __init__(
        self,
        markets: List[MarketData],
        tokenizer: PreTrainedTokenizer,
        cache_dir: str,
        use_cache: bool = False,
        max_news_per_bp: int = 50,
    ):
        super().__init__(cache_dir=cache_dir, use_cache=use_cache)
        self.markets = markets
        self.tokenizer = tokenizer
        self.max_news_per_bp = max_news_per_bp
        self.datapoints = self._load_or_create_datapoints()

    def _compute_hash(self) -> str:
        content = []
        for m in self.markets:
            bp_count = len(m.daily_breakpoints) if m.daily_breakpoints else 0
            content.append(f"{m.market_id}:{bp_count}")
        return hashlib.md5(''.join(content).encode()).hexdigest()

    def _create_datapoints(self) -> List[Dict[str, Any]]:
        datapoints = []
        
        for market in tqdm(self.markets, desc='Processing markets'):
            if not market.daily_breakpoints:
                continue
            
            for bp in market.daily_breakpoints:
                # Skip breakpoints without news/attributions
                news_list = bp.get('news', [])
                attributions = bp.get('attributions', [])
                if not news_list or not attributions:
                    continue
                
                window_history = bp.get('window_history', [])
                target = bp.get('after', {})
                
                # First: Filter out zero-score news and collect valid items
                valid_items = []
                for attr in attributions:
                    news_idx = attr.get('news_idx', 0)
                    score = attr.get('score')
                    score = float(score) if score is not None else 0.0
                    
                    if score <= 0 or news_idx >= len(news_list):
                        continue
                    
                    valid_items.append((news_list[news_idx], score))
                
                # Skip if fewer than 2 valid items
                if len(valid_items) < 2:
                    continue
                
                # Then: Limit to max_news_per_bp (keep top scored)
                if len(valid_items) > self.max_news_per_bp:
                    valid_items.sort(key=lambda x: x[1], reverse=True)
                    valid_items = valid_items[:self.max_news_per_bp]
                
                # Build prompts for each valid news item
                input_ids_list = []
                attention_mask_list = []
                weights_list = []
                
                for news, score in valid_items:
                    prompt = self._build_prompt(market, window_history, target, news)
                    
                    encoding = self.tokenizer(
                        prompt,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors='pt',
                    )
                    
                    input_ids_list.append(encoding['input_ids'].squeeze(0))
                    attention_mask_list.append(encoding['attention_mask'].squeeze(0))
                    weights_list.append(score)
                
                if len(input_ids_list) < 2:
                    continue
                
                # Pad sequences
                input_ids_tensor = pad_sequence(
                    input_ids_list, batch_first=True, 
                    padding_value=self.tokenizer.pad_token_id or 0
                )
                attention_mask_tensor = pad_sequence(
                    attention_mask_list, batch_first=True, padding_value=0
                )
                
                datapoints.append({
                    'input_ids': input_ids_tensor,
                    'attention_mask': attention_mask_tensor,
                    'weights': torch.tensor(weights_list, dtype=torch.float),
                    'label': torch.tensor(target.get('p', 0.5), dtype=torch.float),
                    'market_id': market.market_id,
                    'event_id': market.event_id,
                    't': target.get('t'),
                })
        
        return datapoints

    def _build_prompt(
        self,
        market: MarketData,
        window_history: List[Dict],
        target: Dict,
        news: Dict,
    ) -> str:
        lines = [f'Event: {market.question}']
        if market.description:
            lines.append(f'Description: {market.description}')
        
        # IMPORTANT: Exclude target point from history to prevent data leakage!
        target_ts = target.get('t')
        history_before_target = [
            day for day in window_history 
            if day.get('t') != target_ts
        ]
        
        lines.append('\nRecent price history:')
        for day in history_before_target[-5:]:  # Last 5 days for brevity
            date = unix_to_date(day['t'])
            lines.append(f"  {date}: {day['p']:.3f}")
        
        target_date = unix_to_date(target['t'])
        news_title = news.get('title', '')
        news_desc = news.get('description', '')
        
        lines.append(f'\nNews: {news_title}')
        if news_desc:
            lines.append(f'{news_desc}')
        lines.append(f'\nPredict the probability on {target_date}:')
        
        return '\n'.join(lines)


class PriorAttributerDataset(BaseDataset):
    """
    Simple dataset for training PriorAttributer.
    
    Uses precomputed attributions (from PosteriorAttributer) as targets.
    The model learns to predict attribution scores without seeing news content.
    """
    def __init__(
        self,
        markets: List[MarketData],
        tokenizer: PreTrainedTokenizer,
        cache_dir: str,
        use_cache: bool = False,
        max_news_per_bp: int = 50,
        target_temperature: float = 0.5,  # Lower = sharper distribution, higher = more uniform
    ):
        super().__init__(cache_dir=cache_dir, use_cache=use_cache)
        self.markets = markets
        self.tokenizer = tokenizer
        self.max_news_per_bp = max_news_per_bp
        self.target_temperature = target_temperature
        self.datapoints = self._load_or_create_datapoints()

    def _compute_hash(self) -> str:
        content = []
        for m in self.markets:
            bp_count = len(m.daily_breakpoints) if m.daily_breakpoints else 0
            content.append(f"{m.market_id}:{bp_count}")
        return hashlib.md5(''.join(content).encode()).hexdigest()

    def _create_datapoints(self) -> List[Dict[str, Any]]:
        datapoints = []
        
        for market in tqdm(self.markets, desc='Processing markets'):
            if not market.daily_breakpoints:
                continue
            
            for bp in market.daily_breakpoints:
                news_list = bp.get('news', [])
                attributions = bp.get('attributions', [])
                if not news_list or not attributions:
                    continue
                
                # Ensure attributions match news count
                if len(attributions) != len(news_list):
                    continue
                
                window_history = bp.get('window_history', [])
                target = bp.get('after', {})
                
                # First: Filter out zero-score news items to avoid wasting resources
                filtered_items = []
                for news, attr in zip(news_list, attributions):
                    score = attr.get('score')
                    score = float(score) if score is not None else 0.0
                    if score > 0:
                        filtered_items.append((news, attr, score))
                
                # Skip if fewer than 2 non-zero scores
                if len(filtered_items) < 2:
                    continue
                
                # Then: Limit to max_news_per_bp (keep top scored)
                if len(filtered_items) > self.max_news_per_bp:
                    filtered_items.sort(key=lambda x: x[2], reverse=True)
                    filtered_items = filtered_items[:self.max_news_per_bp]
                
                # Extract filtered news and attributions
                news_list = [x[0] for x in filtered_items]
                attributions = [x[1] for x in filtered_items]
                scores = [x[2] for x in filtered_items]
                    
                scores_tensor = torch.tensor(scores, dtype=torch.float)
                # Use softmax with temperature to create sharper target distribution
                # Lower temperature = more concentrated on top items
                p_dist = F.softmax(scores_tensor / self.target_temperature, dim=0)
                
                # Build prompts for each news item
                # Each prompt includes: question + history + one news article
                all_input_ids = []
                all_attention_masks = []
                
                for news_item in news_list:
                    prompt = self._build_prompt_with_news(market, window_history, target, news_item)
                    encoding = self.tokenizer(
                        prompt,
                        padding='max_length',
                        truncation=True,
                        max_length=512,
                        return_tensors='pt',
                    )
                    all_input_ids.append(encoding['input_ids'])
                    all_attention_masks.append(encoding['attention_mask'])
                
                # Stack to create 2D tensors: (num_news, seq_len)
                input_ids_stacked = torch.cat(all_input_ids, dim=0)
                attention_mask_stacked = torch.cat(all_attention_masks, dim=0)
                
                datapoints.append({
                    'input_ids': input_ids_stacked,  # (num_news, seq_len)
                    'attention_mask': attention_mask_stacked,  # (num_news, seq_len)
                    'p_dist': p_dist,  # Target distribution (num_news,)
                    'news': news_list,  # For reference
                    'label': torch.tensor(target.get('p', 0.5), dtype=torch.float),
                    'market_id': market.market_id,
                    'event_id': market.event_id,
                    't': target.get('t'),
                })
        
        return datapoints

    def _build_prompt_with_news(
        self,
        market: MarketData,
        window_history: List[Dict],
        target: Dict,
        news_item: Dict,
    ) -> str:
        """Build PRIOR prompt - does NOT include target price, only predicts news relevance.
        
        The model learns to predict which news MIGHT affect the market,
        without knowing what actually happened to the price.
        """
        lines = [f'Prediction Market: {market.question}']
        if market.description:
            desc = market.description[:200] + '...' if len(market.description or '') > 200 else market.description
            lines.append(f'Description: {desc}')
        
        # Show recent price history (BEFORE the breakpoint, not after)
        # Exclude the last point if it matches the target (after) timestamp
        target_ts = target.get('t')
        history_before_target = [
            day for day in window_history 
            if day.get('t') != target_ts
        ]
        
        lines.append('\nRecent price history:')
        for day in history_before_target[-5:]:
            date = unix_to_date(day['t'])
            lines.append(f"  {date}: {day['p']:.3f}")
        
        # Target date (when we're predicting for) - but NOT the target price
        target_date = unix_to_date(target['t'])
        lines.append(f'\nPredicting for: {target_date}')
        
        # Add the news article
        lines.append('\nNews article:')
        news_title = news_item.get('title', '')
        news_desc = news_item.get('description', '') or ''
        news_date = news_item.get('published_at', '')
        if news_date:
            lines.append(f'Date: {news_date}')
        lines.append(f'Title: {news_title}')
        if news_desc:
            desc = news_desc[:300] + '...' if len(news_desc) > 300 else news_desc
            lines.append(f'Content: {desc}')
        
        # Prior task: predict relevance WITHOUT knowing price outcome
        lines.append('\nHow relevant is this news to the prediction market outcome?')
        
        return '\n'.join(lines)

    def _build_prompt(
        self,
        market: MarketData,
        window_history: List[Dict],
        target: Dict,
    ) -> str:
        """Build basic prompt without news (legacy, for compatibility)."""
        lines = [f'Event: {market.question}']
        if market.description:
            lines.append(f'Description: {market.description}')
        
        lines.append('\nRecent price history:')
        for day in window_history[-5:]:
            date = unix_to_date(day['t'])
            lines.append(f"  {date}: {day['p']:.3f}")
        
        target_date = unix_to_date(target['t'])
        lines.append(f'\nPredict which news is most relevant for {target_date}:')
        
        return '\n'.join(lines)


class RAGMultiEventForecasterDataset(MultiEventForecasterDataset):
    """
    RAG-enhanced version that includes similar markets in the prompt.
    """
    def __init__(
        self,
        markets: List[MarketData],
        similar_markets: Dict[str, List[MarketData]],
        tokenizer: PreTrainedTokenizer,
        cache_dir: str,
        max_similar: int = 3,
        use_cache: bool = False,
    ):
        self.similar_markets = similar_markets
        self.max_similar = max_similar
        super().__init__(
            markets=markets,
            tokenizer=tokenizer,
            cache_dir=cache_dir,
            use_cache=use_cache,
        )

    def _build_prompt(
        self,
        market: MarketData,
        window_history: List[Dict],
        target: Dict,
        news: Dict,
    ) -> str:
        # Start with base prompt
        lines = [f'Event: {market.question}']
        if market.description:
            lines.append(f'Description: {market.description}')
        
        # Add similar markets context
        sim_markets = self.similar_markets.get(market.market_id, [])[:self.max_similar]
        if sim_markets:
            lines.append('\nSimilar events for reference:')
            for sm in sim_markets:
                lines.append(f'- {sm.question}')
                if sm.outcome:
                    lines.append(f'  Outcome: {sm.outcome}')
        
        # IMPORTANT: Exclude target point from history to prevent data leakage!
        target_ts = target.get('t')
        history_before_target = [
            day for day in window_history 
            if day.get('t') != target_ts
        ]
        
        lines.append('\nRecent price history:')
        for day in history_before_target[-5:]:
            date = unix_to_date(day['t'])
            lines.append(f"  {date}: {day['p']:.3f}")
        
        target_date = unix_to_date(target['t'])
        news_title = news.get('title', '')
        news_desc = news.get('description', '')
        
        lines.append(f'\nNews: {news_title}')
        if news_desc:
            lines.append(f'{news_desc}')
        lines.append(f'\nPredict the probability on {target_date}:')
        
        return '\n'.join(lines)
