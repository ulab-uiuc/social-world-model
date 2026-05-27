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
        max_seq_length: int = 1024,
        max_history_len: int = None,  # Max number of history points to use (None = use all)
        predict_absolute_price: bool = False,  # True for old checkpoints that predict absolute price
        min_positive_attributions: int = 0,  # Minimum positive attributions to include sample
        weight_temperature: float = 1.0,  # Temperature for softmax on attribution weights (lower = sharper)
        min_max_attribution_score: float = 0.0,  # Only include breakpoints where max attribution score >= this
        news_score_threshold: float = 0.0,  # Per-news threshold; drop news with score < this from input
        null_subsample_ratio: float = 1.0,  # Keep this fraction of null events (1.0=all, <1.0 to rebalance)
        null_subsample_seed: int = 42,
        only_null: bool = False,  # If True, KEEP only null events (drop all has-news). For time-series LM baseline.
        window_std_threshold: float = 0.0,  # Drop events where pre-BP window price std < this (illiquid market filter; 0.02 recommended).
        null_max_delta: float = 0.0,  # If >0, drop null events with |Δp| > this (filters retrieval-fail noise; 0.05 recommended).
        null_exclude_breakpoint: bool = False,  # Drop null events whose sample_type=='breakpoint' (oracle false-negative suspects).
        flip_augment: bool = False,  # For each datapoint, also emit a 1-p mirror copy (debias via symmetry).
    ):
        super().__init__(cache_dir=cache_dir, use_cache=use_cache)
        self.markets = markets
        self.tokenizer = tokenizer
        self.max_news_per_bp = max_news_per_bp
        self.max_seq_length = max_seq_length
        self.max_history_len = max_history_len
        self.predict_absolute_price = predict_absolute_price
        self.min_positive_attributions = min_positive_attributions
        self.weight_temperature = weight_temperature
        self.min_max_attribution_score = min_max_attribution_score
        self.news_score_threshold = news_score_threshold
        self.null_subsample_ratio = null_subsample_ratio
        self.null_subsample_seed = null_subsample_seed
        self.only_null = only_null
        self.window_std_threshold = window_std_threshold
        self.null_max_delta = null_max_delta
        self.null_exclude_breakpoint = null_exclude_breakpoint
        self.flip_augment = flip_augment
        self.datapoints = self._load_or_create_datapoints()

    def _compute_hash(self) -> str:
        content = []
        for m in self.markets:
            bp_count = len(m.daily_breakpoints) if m.daily_breakpoints else 0
            content.append(f"{m.market_id}:{bp_count}")
        content.append(f"min_attr:{self.min_positive_attributions}")
        content.append(f"min_max_score:{self.min_max_attribution_score}")
        content.append(f"news_thr:{self.news_score_threshold}")
        content.append(f"null_ratio:{self.null_subsample_ratio}:{self.null_subsample_seed}")
        content.append(f"only_null:{self.only_null}")
        content.append(f"window_std:{self.window_std_threshold}")
        content.append(f"null_max_delta:{self.null_max_delta}")
        content.append(f"null_exclude_bp:{self.null_exclude_breakpoint}")
        content.append(f"flip_aug:{self.flip_augment}")
        return hashlib.md5(''.join(content).encode()).hexdigest()

    def _create_datapoints(self) -> List[Dict[str, Any]]:
        datapoints = []
        import random as _random
        _rng = _random.Random(self.null_subsample_seed)
        _null_kept = _null_dropped = _has_kept = 0

        for market in tqdm(self.markets, desc='Processing markets'):
            if not market.daily_breakpoints:
                continue

            for bp in market.daily_breakpoints:
                # Skip if no news list
                news_list = bp.get('news', [])
                # Prefer V2 direction-aware attributions, fallback to V1
                attributions = bp.get('attributions_v2', bp.get('attributions', []))
                if not news_list:
                    continue

                window_history = bp.get('window_history', [])
                before = bp.get('before', {})
                target = bp.get('after', {})

                before_price = before.get('p', 0.5)
                after_price = target.get('p', 0.5)
                price_delta = after_price - before_price

                # Liquidity filter: drop events whose pre-BP price window was
                # too stable (illiquid market — small moves are likely thin-book
                # noise rather than real news reactions). Computed BEFORE the
                # bp pair (last 2 entries of window_history).
                if self.window_std_threshold > 0:
                    pre_bp_prices = [h.get('p', 0) for h in window_history[:-2]]
                    if len(pre_bp_prices) >= 5:
                        import statistics as _st
                        if _st.stdev(pre_bp_prices) < self.window_std_threshold:
                            continue
                    elif len(pre_bp_prices) > 0:
                        # Not enough history → also drop (conservative)
                        continue

                # Filter by max attribution score threshold
                if self.min_max_attribution_score > 0 and attributions:
                    all_scores = [float(a.get('relevance', a.get('score', 0)) or 0) for a in attributions if a.get('relevance') is not None or a.get('score') is not None]
                    if not all_scores or max(all_scores) < self.min_max_attribution_score:
                        continue

                # Collect news items with positive scores
                related_items = []
                had_any_positive_news = False
                for attr in attributions:
                    news_idx = attr.get('news_idx', 0)
                    # Use relevance (V2) or score (V1)
                    score = attr.get('relevance', attr.get('score'))
                    score = float(score) if score is not None else 0.0
                    # Extract direction from V2 attributions (-1, 0, +1)
                    direction = attr.get('direction', 0)

                    if score <= 0 or news_idx >= len(news_list):
                        continue
                    had_any_positive_news = True
                    # Per-news score threshold: drop weak/noisy news from input
                    if score < self.news_score_threshold:
                        continue

                    related_items.append((news_list[news_idx], score, direction))

                # Drop events that originally had news but lost all of it to
                # the threshold filter — don't pollute training as "null" (their
                # target Δp is non-zero by oracle assessment, no_news prompt
                # would give conflicting signal).
                if self.news_score_threshold > 0 and had_any_positive_news and not related_items:
                    continue

                # Filter by minimum positive attributions
                if self.min_positive_attributions > 0 and len(related_items) < self.min_positive_attributions:
                    continue

                if self.predict_absolute_price and not self.only_null:
                    # Old format: require at least 2 valid items, no "no news" option.
                    # Skip this filter in only_null mode (we want zero-news events).
                    if len(related_items) < 2:
                        continue

                # Limit to max_news_per_bp (keep top scored)
                if len(related_items) > self.max_news_per_bp:
                    related_items.sort(key=lambda x: x[1], reverse=True)
                    related_items = related_items[:self.max_news_per_bp]

                # Rebalance: subsample null events (no positive attributions) to
                # counter forecaster's regression-to-mean bias. Has-news kept always.
                is_null = (len(related_items) == 0)
                if self.only_null and not is_null:
                    continue  # only_null mode: drop has-news events (time-series LM baseline)
                # Clean-null filters: when training on null events, drop
                # retrieval-failure suspects (sample_type=breakpoint) and big-Δ
                # outliers (|Δp| > null_max_delta).
                if is_null and self.null_exclude_breakpoint and bp.get('sample_type') == 'breakpoint':
                    continue
                if is_null and self.null_max_delta > 0 and abs(price_delta) > self.null_max_delta:
                    continue
                if is_null and self.null_subsample_ratio < 1.0:
                    if _rng.random() >= self.null_subsample_ratio:
                        _null_dropped += 1
                        continue
                    _null_kept += 1
                elif is_null:
                    _null_kept += 1
                else:
                    _has_kept += 1

                # Build prompts for each valid news item
                input_ids_list = []
                attention_mask_list = []
                weights_list = []

                for news, score, direction in related_items:
                    prompt = self._build_prompt(market, window_history, target, news, direction=direction)
                    encoding = self.tokenizer(
                        prompt,
                        padding=True,
                        truncation=True,
                        max_length=self.max_seq_length,
                        return_tensors='pt',
                    )
                    input_ids_list.append(encoding['input_ids'].squeeze(0))
                    attention_mask_list.append(encoding['attention_mask'].squeeze(0))
                    weights_list.append(score)

                # In only_null + predict_absolute_price mode, all events are
                # null → need to add the no_news_prompt as the single input.
                if self.predict_absolute_price and self.only_null and len(related_items) == 0:
                    no_news_prompt = self._build_no_news_prompt(market, window_history, target)
                    no_news_encoding = self.tokenizer(
                        no_news_prompt, padding=True, truncation=True,
                        max_length=self.max_seq_length, return_tensors='pt',
                    )
                    input_ids_list.append(no_news_encoding['input_ids'].squeeze(0))
                    attention_mask_list.append(no_news_encoding['attention_mask'].squeeze(0))
                    weights_list.append(1.0)
                if not self.predict_absolute_price:
                    if len(related_items) > 1:
                        # Multiple news: add "no news" baseline option
                        no_news_prompt = self._build_no_news_prompt(market, window_history, target)
                        no_news_encoding = self.tokenizer(
                            no_news_prompt,
                            padding=True,
                            truncation=True,
                            max_length=self.max_seq_length,
                            return_tensors='pt',
                        )
                        input_ids_list.append(no_news_encoding['input_ids'].squeeze(0))
                        attention_mask_list.append(no_news_encoding['attention_mask'].squeeze(0))
                        weights_list.append(0.001)
                    elif len(related_items) == 0:
                        # No positive attributions: use "no news" prompt for history-only prediction
                        no_news_prompt = self._build_no_news_prompt(market, window_history, target)
                        no_news_encoding = self.tokenizer(
                            no_news_prompt,
                            padding=True,
                            truncation=True,
                            max_length=self.max_seq_length,
                            return_tensors='pt',
                        )
                        input_ids_list.append(no_news_encoding['input_ids'].squeeze(0))
                        attention_mask_list.append(no_news_encoding['attention_mask'].squeeze(0))
                        weights_list.append(1.0)

                if not input_ids_list:
                    continue

                weights_tensor = torch.tensor(weights_list, dtype=torch.float)
                if not self.predict_absolute_price and len(weights_list) > 1:
                    # Softmax weights only when multiple items
                    weights_tensor = F.softmax(weights_tensor / self.weight_temperature, dim=0)
                elif not self.predict_absolute_price:
                    # Single item: weight = 1.0 (direct regression)
                    weights_tensor = torch.ones_like(weights_tensor)

                # Pad sequences
                input_ids_tensor = pad_sequence(
                    input_ids_list, batch_first=True,
                    padding_value=self.tokenizer.pad_token_id or 0
                )
                attention_mask_tensor = pad_sequence(
                    attention_mask_list, batch_first=True, padding_value=0
                )

                # Label: absolute price for old checkpoints, delta for new
                label = after_price if self.predict_absolute_price else price_delta

                datapoints.append({
                    'input_ids': input_ids_tensor,
                    'attention_mask': attention_mask_tensor,
                    'weights': weights_tensor,
                    'label': torch.tensor(label, dtype=torch.float),
                    'before_price': torch.tensor(before_price, dtype=torch.float),
                    'after_price': torch.tensor(after_price, dtype=torch.float),
                    'market_id': market.market_id,
                    'event_id': market.event_id,
                    't': target.get('t'),
                    'is_null': is_null,
                })

                # === FLIP AUGMENTATION ===
                # Binary markets: p and 1-p are mathematically symmetric.
                # Emit a mirror datapoint with all prices flipped (p → 1-p).
                # This debias training (forces mean Δp → 0 by construction).
                if self.flip_augment:
                    # Build flipped versions of window_history and target dicts
                    flip_hist = [{'t': h['t'], 'p': 1.0 - h.get('p', 0.5)} for h in window_history]
                    flip_target = {'t': target.get('t'), 'p': 1.0 - after_price}
                    flip_bef_price = 1.0 - before_price
                    flip_aft_price = 1.0 - after_price
                    flip_label = flip_aft_price if self.predict_absolute_price else (flip_aft_price - flip_bef_price)
                    # Rebuild prompts with flipped prices (news content unchanged — abstract symmetry)
                    flip_ids_list, flip_am_list = [], []
                    for news, score, direction in related_items:
                        p2 = self._build_prompt(market, flip_hist, flip_target, news, direction=direction)
                        e2 = self.tokenizer(p2, padding=True, truncation=True,
                                            max_length=self.max_seq_length, return_tensors='pt')
                        flip_ids_list.append(e2['input_ids'].squeeze(0))
                        flip_am_list.append(e2['attention_mask'].squeeze(0))
                    if self.predict_absolute_price and self.only_null and len(related_items) == 0:
                        p2 = self._build_no_news_prompt(market, flip_hist, flip_target)
                        e2 = self.tokenizer(p2, padding=True, truncation=True,
                                            max_length=self.max_seq_length, return_tensors='pt')
                        flip_ids_list.append(e2['input_ids'].squeeze(0))
                        flip_am_list.append(e2['attention_mask'].squeeze(0))
                    if not self.predict_absolute_price:
                        if len(related_items) > 1:
                            p2 = self._build_no_news_prompt(market, flip_hist, flip_target)
                            e2 = self.tokenizer(p2, padding=True, truncation=True,
                                                max_length=self.max_seq_length, return_tensors='pt')
                            flip_ids_list.append(e2['input_ids'].squeeze(0))
                            flip_am_list.append(e2['attention_mask'].squeeze(0))
                        elif len(related_items) == 0:
                            p2 = self._build_no_news_prompt(market, flip_hist, flip_target)
                            e2 = self.tokenizer(p2, padding=True, truncation=True,
                                                max_length=self.max_seq_length, return_tensors='pt')
                            flip_ids_list.append(e2['input_ids'].squeeze(0))
                            flip_am_list.append(e2['attention_mask'].squeeze(0))
                    if flip_ids_list:
                        flip_ids = pad_sequence(flip_ids_list, batch_first=True,
                                                padding_value=self.tokenizer.pad_token_id or 0)
                        flip_am = pad_sequence(flip_am_list, batch_first=True, padding_value=0)
                        datapoints.append({
                            'input_ids': flip_ids,
                            'attention_mask': flip_am,
                            'weights': weights_tensor,
                            'label': torch.tensor(flip_label, dtype=torch.float),
                            'before_price': torch.tensor(flip_bef_price, dtype=torch.float),
                            'after_price': torch.tensor(flip_aft_price, dtype=torch.float),
                            'market_id': market.market_id + '__flip',
                            'event_id': market.event_id,
                            't': target.get('t'),
                            'is_null': is_null,
                        })

        if self.null_subsample_ratio < 1.0 or _null_dropped > 0:
            tot = _null_kept + _has_kept
            print(f'[MultiEventForecasterDataset] null_subsample_ratio={self.null_subsample_ratio} '
                  f'→ has_news={_has_kept} ({100*_has_kept/max(tot,1):.1f}%)  '
                  f'null_kept={_null_kept} ({100*_null_kept/max(tot,1):.1f}%)  '
                  f'null_dropped={_null_dropped}')

        return datapoints

    def _build_prompt(
        self,
        market: MarketData,
        window_history: List[Dict],
        target: Dict,
        news: Dict,
        direction: int = 0,
    ) -> str:
        lines = [f'Event: {market.question}']
        if market.description:
            lines.append(f'Description: {market.description}')

        target_ts = target.get('t')

        if self.predict_absolute_price:
            # Old format: exclude only exact target timestamp, show last 5 days
            history_before_target = [
                day for day in window_history
                if day.get('t') != target_ts
            ]
            history_before_target = history_before_target[-5:]
        else:
            # New format: exclude target and future, optionally truncate
            history_before_target = [
                day for day in window_history
                if day.get('t') < target_ts
            ]
            if self.max_history_len is not None and len(history_before_target) > self.max_history_len:
                history_before_target = history_before_target[-self.max_history_len:]

        lines.append('\nRecent price history:')
        for day in history_before_target:
            date = unix_to_date(day['t'])
            lines.append(f"  {date}: {day['p']:.3f}")

        target_date = unix_to_date(target['t'])
        news_title = news.get('title', '')
        news_desc = news.get('description', '')

        # Add direction tag from V2 attribution if available
        dir_tag = ''
        if direction == 1:
            dir_tag = '[INCREASES PROBABILITY] '
        elif direction == -1:
            dir_tag = '[DECREASES PROBABILITY] '

        lines.append(f'\nNews: {dir_tag}{news_title}')
        if news_desc:
            lines.append(f'{news_desc}')
        lines.append(f'\nPredict the probability on {target_date}:')

        return '\n'.join(lines)

    def _build_no_news_prompt(
        self,
        market: MarketData,
        window_history: List[Dict],
        target: Dict,
    ) -> str:
        """Build prompt for "no news" option - predict based on history only."""
        lines = [f'Event: {market.question}']
        if market.description:
            lines.append(f'Description: {market.description}')
        
        # IMPORTANT: Exclude target point AND future points from history to prevent data leakage!
        target_ts = target.get('t')
        history_before_target = [
            day for day in window_history 
            if day.get('t') < target_ts
        ]
        
        # Truncate history if max_history_len is set (keep most recent)
        if self.max_history_len is not None and len(history_before_target) > self.max_history_len:
            history_before_target = history_before_target[-self.max_history_len:]
        
        lines.append('\nRecent price history:')
        for day in history_before_target:
            date = unix_to_date(day['t'])
            lines.append(f"  {date}: {day['p']:.3f}")
        
        target_date = unix_to_date(target['t'])
        
        lines.append(f'\nNews: No relevant news.')
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
        max_seq_length: int = 1024,
        target_temperature: float = 0.5,  # Lower = sharper distribution, higher = more uniform
        use_raw_scores: bool = False,  # If True, use raw scores (for MSE loss) instead of softmax
        include_negatives: bool = True,
    ):
        super().__init__(cache_dir=cache_dir, use_cache=use_cache)
        self.markets = markets
        self.tokenizer = tokenizer
        self.max_news_per_bp = max_news_per_bp
        self.max_seq_length = max_seq_length
        self.target_temperature = target_temperature
        self.use_raw_scores = use_raw_scores
        # When True, has-event training groups include oracle-NEGATIVE news
        # (score=0) alongside positives, so the model learns causal-vs-noncausal
        # discrimination — matching what attribute_breakpoint scores at inference.
        # The legacy positive-only behavior produced AUC≈0.5 at every model size.
        self.include_negatives = include_negatives
        self.datapoints = self._load_or_create_datapoints()

    def _compute_hash(self) -> str:
        content = [f"include_negatives:{int(self.include_negatives)}"]
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
                if not news_list:
                    continue
                
                window_history = bp.get('window_history', [])
                target = bp.get('after', {})
                
                # Map news_idx -> oracle score (positives only).
                pos_score = {}
                for attr in attributions:
                    news_idx = attr.get('news_idx', 0)
                    score = attr.get('score')
                    score = float(score) if score is not None else 0.0
                    if score <= 0 or not (0 <= news_idx < len(news_list)):
                        continue
                    pos_score[news_idx] = score

                is_null_event = len(pos_score) == 0

                # Build prompts for each news item
                input_ids_list = []
                attention_mask_list = []
                weights_list = []

                if is_null_event:
                    # Null event (normal_point): include a sample of news with score=0
                    # so the model learns "all these news are irrelevant"
                    null_news = news_list[:self.max_news_per_bp]
                    null_score = 0.001  # near-zero score for all news
                    for news in null_news:
                        prompt = self._build_prompt_with_news(market, window_history, target, news)
                        encoding = self.tokenizer(
                            prompt,
                            padding=True,
                            truncation=True,
                            max_length=self.max_seq_length,
                            return_tensors='pt',
                        )
                        input_ids_list.append(encoding['input_ids'].squeeze(0))
                        attention_mask_list.append(encoding['attention_mask'].squeeze(0))
                        weights_list.append(null_score)
                else:
                    # Has event (breakpoint): include BOTH positives (oracle score)
                    # AND oracle-negative news (score=0) up to max_news_per_bp, so the
                    # softmax / mse_rank ranking loss actually teaches positive>negative
                    # — matching what attribute_breakpoint scores at inference (all
                    # news). Without negatives the model never learns the causal-vs-
                    # noncausal discrimination → AUC≈0.5 at every size.
                    # Order positives by oracle score desc so the highest-signal
                    # positives are kept under a tight cap.
                    pos_idxs = sorted(pos_score.keys(), key=lambda i: -pos_score[i])
                    if self.include_negatives:
                        neg_idxs = [i for i in range(len(news_list)) if i not in pos_score]
                        # Reserve up to half the cap for negatives whenever any
                        # exist, so the group is GUARANTEED mixed and the ranking
                        # / softmax loss always sees a pos-vs-neg contrast (the
                        # whole point of the fix). Without this, breakpoints
                        # with many oracle positives fill all slots with
                        # positives and the discrimination signal vanishes again.
                        n_max = self.max_news_per_bp
                        if neg_idxs:
                            n_neg_take = min(len(neg_idxs), max(1, n_max // 2))
                            n_pos_take = n_max - n_neg_take
                            use_idxs = pos_idxs[:n_pos_take] + neg_idxs[:n_neg_take]
                        else:
                            use_idxs = pos_idxs[:n_max]
                    else:
                        use_idxs = pos_idxs[:self.max_news_per_bp]
                    for i in use_idxs:
                        news = news_list[i]
                        score = pos_score.get(i, 0.0)
                        prompt = self._build_prompt_with_news(market, window_history, target, news)
                        encoding = self.tokenizer(
                            prompt,
                            padding=True,
                            truncation=True,
                            max_length=self.max_seq_length,
                            return_tensors='pt',
                        )
                        input_ids_list.append(encoding['input_ids'].squeeze(0))
                        attention_mask_list.append(encoding['attention_mask'].squeeze(0))
                        weights_list.append(score)

                # Always add "no news" option
                # For null events: no_news gets high weight (it's the correct answer)
                # For breakpoints: no_news gets low weight
                no_news_weight = 1.0 if is_null_event else 0.001
                no_news_prompt = self._build_no_news_prompt(market, window_history, target)
                no_news_encoding = self.tokenizer(
                    no_news_prompt,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors='pt',
                )
                input_ids_list.append(no_news_encoding['input_ids'].squeeze(0))
                attention_mask_list.append(no_news_encoding['attention_mask'].squeeze(0))
                weights_list.append(no_news_weight)
                
                weights_tensor = torch.tensor(weights_list, dtype=torch.float)
                if self.use_raw_scores:
                    # Raw scores for MSE/ranking loss — no softmax. Per-news
                    # targets: positives = oracle score, negatives = 0, no-news
                    # = 0.001 (null) or 1.0 (null event). MSERankTrainer reads
                    # these and applies sigmoid+MSE + a margin ranking loss
                    # between target>0.05 and target<=0.05; with negatives
                    # now present that ranking loss fires per group.
                    p_dist = weights_tensor
                else:
                    # KL target. Make it separable in BOTH directions:
                    #   null  -> all mass on the no-news option (last element)
                    #   has-news -> all mass on POSITIVES (proportional to their
                    #               oracle scores); negatives and no-news ~= 0
                    # The previous code softmaxed [pos_scores..., 0,0,...,no_news]
                    # which with negatives present makes positives barely beat
                    # negatives (uniform-ish target); the model then can't learn
                    # the causal-vs-noncausal discrimination.
                    eps = 1e-4
                    n = len(weights_list)
                    if is_null_event:
                        p_dist = torch.full((n,), eps / max(n - 1, 1))
                        p_dist[-1] = 1.0 - eps
                    else:
                        news_w = weights_tensor[:-1].clamp(min=0.0)
                        total = news_w.sum()
                        if total > 0:
                            news_p = news_w / total * (1.0 - eps)
                        else:
                            news_p = torch.full_like(news_w, (1.0 - eps) / max(len(news_w), 1))
                        p_dist = torch.cat(
                            [news_p, torch.tensor([eps], dtype=torch.float)]
                        )
                    p_dist = p_dist / p_dist.sum()
                
                # Pad sequences (same as MultiEventForecasterDataset)
                input_ids_tensor = pad_sequence(
                    input_ids_list, batch_first=True, 
                    padding_value=self.tokenizer.pad_token_id or 0
                )
                attention_mask_tensor = pad_sequence(
                    attention_mask_list, batch_first=True, padding_value=0
                )
                
                datapoints.append({
                    'input_ids': input_ids_tensor,  # (num_news, seq_len)
                    'attention_mask': attention_mask_tensor,  # (num_news, seq_len)
                    'p_dist': p_dist,  # Target distribution (num_news,)
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
        """Build PRIOR prompt — predicts news relevance without target price.

        Order matters: news content goes FIRST so right-truncation removes
        less-important context (price history, description) before touching
        the news. Previously the news came last and tokenizer truncation
        (max_seq_length=512) chopped 71-206 tokens off the END of every
        sample — i.e. the rating question AND part of the news content
        were always cut. The model never actually saw the discriminative
        signal, which is why AUC sat at 0.5 regardless of size / loss.
        """
        news_title = news_item.get('title', '')
        news_desc = news_item.get('description', '') or ''
        news_date = news_item.get('published_at', '')
        target_ts = target.get('t')
        target_date = unix_to_date(target['t'])

        lines = ['News article:']
        if news_date:
            lines.append(f'Date: {news_date}')
        lines.append(f'Title: {news_title}')
        if news_desc:
            lines.append(f'Content: {news_desc}')

        lines.append(f'\nPrediction Market: {market.question}')
        if market.description:
            desc = market.description[:200] + '...' if len(market.description or '') > 200 else market.description
            lines.append(f'Description: {desc}')

        lines.append(f'\nPredicting for: {target_date}')
        history_before_target = [
            day for day in window_history if day.get('t') < target_ts
        ]
        if history_before_target:
            lines.append('Recent price history:')
            for day in history_before_target:
                date = unix_to_date(day['t'])
                lines.append(f"  {date}: {day['p']:.3f}")

        lines.append('\nDoes this news have a causal relationship with the price change of this prediction market? Rate higher only if the news could directly cause the market price to move. News that is merely topically related but would not causally drive a price change should receive a low score.')
        
        return '\n'.join(lines)

    def _build_no_news_prompt(
        self,
        market: MarketData,
        window_history: List[Dict],
        target: Dict,
    ) -> str:
        """Build PRIOR prompt for "no news" option."""
        lines = [f'Prediction Market: {market.question}']
        if market.description:
            desc = market.description[:200] + '...' if len(market.description or '') > 200 else market.description
            lines.append(f'Description: {desc}')
        
        # Show recent price history (BEFORE the breakpoint, not after)
        target_ts = target.get('t')
        history_before_target = [
            day for day in window_history 
            if day.get('t') < target_ts
        ]
        
        lines.append('\nRecent price history:')
        for day in history_before_target:
            date = unix_to_date(day['t'])
            lines.append(f"  {date}: {day['p']:.3f}")
        
        target_date = unix_to_date(target['t'])
        lines.append(f'\nPredicting for: {target_date}')
        
        lines.append('\nNews article: No relevant news available.')
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
        for day in window_history:
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
        max_seq_length: int = 1024,
        use_cache: bool = False,
    ):
        self.similar_markets = similar_markets
        self.max_similar = max_similar
        super().__init__(
            markets=markets,
            tokenizer=tokenizer,
            cache_dir=cache_dir,
            max_seq_length=max_seq_length,
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
            if day.get('t') < target_ts
        ]
        
        lines.append('\nRecent price history:')
        for day in history_before_target:
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
