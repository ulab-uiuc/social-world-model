"""Dataset classes for v6 records.

One v6 record == one training example. See data/v6/README.md for schema.
"""
import random
import statistics
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from .data import Record
from .utils.utils import unix_to_date


NULL_SUBSAMPLE_SEED = 42


def _pack_prompts(
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    max_seq_length: int,
) -> Dict[str, torch.Tensor]:
    """Tokenize a list of prompts and pad them into a single (N, L) batch."""
    ids_list, mask_list = [], []
    for p in prompts:
        enc = tokenizer(
            p, padding=True, truncation=True,
            max_length=max_seq_length, return_tensors='pt',
        )
        ids_list.append(enc['input_ids'].squeeze(0))
        mask_list.append(enc['attention_mask'].squeeze(0))
    return {
        'input_ids': pad_sequence(
            ids_list, batch_first=True,
            padding_value=tokenizer.pad_token_id or 0,
        ),
        'attention_mask': pad_sequence(mask_list, batch_first=True, padding_value=0),
    }


class MultiEventForecasterDataset(Dataset):
    """Forecaster training set: predict target.p from (question, history, news).

    Each record is one training group. We emit one prompt per
    positive-attribution news (or a single no-news prompt for null records)
    with a softmax weight vector; WeightedTrainer turns those into a
    weighted-mean prediction against target.p.
    """

    def __init__(
        self,
        records: List[Record],
        tokenizer: PreTrainedTokenizer,
        max_news: int = 50,
        max_seq_length: int = 1024,
        window_std_threshold: float = 0.0,
        null_subsample_ratio: float = 1.0,
    ):
        super().__init__()
        self.records = records
        self.tokenizer = tokenizer
        self.max_news = max_news
        self.max_seq_length = max_seq_length
        self.window_std_threshold = window_std_threshold
        self.null_subsample_ratio = null_subsample_ratio
        self.datapoints = self._create_datapoints()
        print(f'Created {len(self.datapoints)} datapoints')

    def __len__(self) -> int:
        return len(self.datapoints)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.datapoints[idx]

    def _create_datapoints(self) -> List[Dict[str, Any]]:
        datapoints: List[Dict[str, Any]] = []
        rng = random.Random(NULL_SUBSAMPLE_SEED)
        null_kept = null_dropped = has_kept = 0

        for record in tqdm(self.records, desc='Building forecaster datapoints'):
            if not record.target or not record.news:
                continue
            if not self._passes_window_std(record.history):
                continue

            items = self._select_items(record, rng)
            if items is None:
                null_dropped += 1
                continue
            if items[0][0] is None:
                null_kept += 1
            else:
                has_kept += 1

            target = record.target
            prompts = [self._build_prompt(record, target, news) for news, _ in items]
            weights = torch.tensor([w for _, w in items], dtype=torch.float)
            if weights.numel() > 1:
                weights = F.softmax(weights, dim=0)
            else:
                weights = torch.ones_like(weights)

            before_p = (
                float(record.history[-1].get('p', 0.5)) if record.history
                else float(target.get('p', 0.5))
            )
            datapoints.append({
                **_pack_prompts(self.tokenizer, prompts, self.max_seq_length),
                'weights': weights,
                'label': torch.tensor(float(target.get('p', 0.5)), dtype=torch.float),
                'before_price': torch.tensor(before_p, dtype=torch.float),
                'market_id': record.market_id,
                'event_id': record.event_id,
                't': target.get('t'),
            })

        if self.null_subsample_ratio < 1.0:
            tot = null_kept + has_kept
            print(f'[MultiEventForecasterDataset] null_subsample_ratio={self.null_subsample_ratio} '
                  f'→ has_news={has_kept} ({100 * has_kept / max(tot, 1):.1f}%)  '
                  f'null_kept={null_kept} ({100 * null_kept / max(tot, 1):.1f}%)  '
                  f'null_dropped={null_dropped}')

        return datapoints

    def _select_items(
        self, record: Record, rng: random.Random,
    ) -> Optional[List]:
        """List of (news_or_None, weight) tuples for this record, or None to drop."""
        positives = self._top_positives(record)
        if positives:
            return positives
        if rng.random() >= self.null_subsample_ratio:
            return None
        return [(None, 1.0)]

    def _top_positives(self, record: Record) -> List:
        n = len(record.news)
        positives = [
            (record.news[a['news_idx']], float(a['score']))
            for a in record.attributions
            if 0 <= a.get('news_idx', -1) < n
            and float(a.get('score') or 0) > 0
        ]
        positives.sort(key=lambda p: -p[1])
        return positives[:self.max_news]

    def _passes_window_std(self, history: List[Dict[str, float]]) -> bool:
        if self.window_std_threshold <= 0:
            return True
        prices = [float(h.get('p', 0)) for h in history]
        return len(prices) >= 5 and statistics.stdev(prices) >= self.window_std_threshold

    def _build_prompt(
        self,
        record: Record,
        target: Dict[str, float],
        news: Optional[Dict[str, Any]],
    ) -> str:
        lines = [f'Event: {record.question}']
        if record.description:
            lines.append(f'Description: {record.description}')

        lines.append('\nRecent price history:')
        for day in record.history:
            lines.append(f"  {unix_to_date(day['t'])}: {day['p']:.3f}")

        if news is None:
            lines.append('\nNews: No relevant news.')
        else:
            lines.append(f'\nNews: {news.get("title", "")}')
            if news.get('description'):
                lines.append(news['description'])
        target_date = unix_to_date(target['t'])
        lines.append(f'\nPredict the probability on {target_date}:')
        return '\n'.join(lines)


class PriorAttributerDataset(Dataset):
    """Attributer training set: KL target over (news ∪ {no-news}).

    Has-news records put mass on positive-attribution news proportional to
    oracle score, with sampled negatives at 0; null records put mass=1 on the
    no-news prompt and 0 on all news.
    """

    def __init__(
        self,
        records: List[Record],
        tokenizer: PreTrainedTokenizer,
        max_news: int = 50,
        max_seq_length: int = 1024,
        include_negatives: bool = True,
        null_subsample_ratio: float = 1.0,
    ):
        super().__init__()
        self.records = records
        self.tokenizer = tokenizer
        self.max_news = max_news
        self.max_seq_length = max_seq_length
        self.include_negatives = include_negatives
        self.null_subsample_ratio = null_subsample_ratio
        self.datapoints = self._create_datapoints()
        print(f'Created {len(self.datapoints)} datapoints')

    def __len__(self) -> int:
        return len(self.datapoints)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.datapoints[idx]

    def _create_datapoints(self) -> List[Dict[str, Any]]:
        datapoints: List[Dict[str, Any]] = []
        rng = random.Random(NULL_SUBSAMPLE_SEED)
        null_kept = null_dropped = has_kept = 0
        for record in tqdm(self.records, desc='Building attributer datapoints'):
            if not record.news or not record.target:
                continue

            target = record.target
            scores_by_idx = self._positive_scores(record)
            is_null = not scores_by_idx
            if is_null:
                if rng.random() >= self.null_subsample_ratio:
                    null_dropped += 1
                    continue
                null_kept += 1
            else:
                has_kept += 1

            use_idxs = self._select_news_idxs(record, scores_by_idx)
            score_list = [scores_by_idx.get(i, 0.0) for i in use_idxs]
            prompts = [self._build_prompt_with_news(record, target, record.news[i])
                       for i in use_idxs]

            prompts.append(self._build_no_news_prompt(record, target))
            score_list.append(1.0 if is_null else 0.0)

            scores = torch.tensor(score_list, dtype=torch.float).clamp(min=0.0)
            total = scores.sum()
            p_dist = scores / total if total > 0 else torch.full_like(scores, 1.0 / scores.size(0))

            datapoints.append({
                **_pack_prompts(self.tokenizer, prompts, self.max_seq_length),
                'p_dist': p_dist,
                'market_id': record.market_id,
                'event_id': record.event_id,
                't': target.get('t'),
            })

        if self.null_subsample_ratio < 1.0:
            tot = null_kept + has_kept
            print(f'[PriorAttributerDataset] null_subsample_ratio={self.null_subsample_ratio} '
                  f'→ has_news={has_kept} ({100 * has_kept / max(tot, 1):.1f}%)  '
                  f'null_kept={null_kept} ({100 * null_kept / max(tot, 1):.1f}%)  '
                  f'null_dropped={null_dropped}')
        return datapoints

    def _positive_scores(self, record: Record) -> Dict[int, float]:
        n = len(record.news)
        return {
            a['news_idx']: float(a['score'])
            for a in record.attributions
            if 0 <= a.get('news_idx', -1) < n
            and float(a.get('score') or 0) > 0
        }

    def _select_news_idxs(
        self,
        record: Record,
        scores_by_idx: Dict[int, float],
    ) -> List[int]:
        n = len(record.news)
        if not scores_by_idx:
            return list(range(min(n, self.max_news)))
        pos_idxs = sorted(scores_by_idx, key=lambda i: -scores_by_idx[i])
        if not self.include_negatives:
            return pos_idxs[:self.max_news]
        neg_idxs = [i for i in range(n) if i not in scores_by_idx]
        if not neg_idxs:
            return pos_idxs[:self.max_news]
        n_neg = min(len(neg_idxs), max(1, self.max_news // 2))
        n_pos = self.max_news - n_neg
        return pos_idxs[:n_pos] + neg_idxs[:n_neg]

    def _build_prompt_with_news(
        self,
        record: Record,
        target: Dict[str, float],
        news: Dict[str, Any],
    ) -> str:
        # News-first ordering: protect the discriminative tail from
        # right-truncation at max_seq_length (matches attributer.py inference).
        target_date = unix_to_date(target['t'])

        lines = ['News article:']
        if news.get('published_at'):
            lines.append(f"Date: {news['published_at']}")
        lines.append(f'Title: {news.get("title", "")}')
        if news.get('description'):
            lines.append(f"Content: {news['description']}")

        lines.append(f'\nPrediction Market: {record.question}')
        if record.description:
            desc = record.description
            if len(desc) > 200:
                desc = desc[:200] + '...'
            lines.append(f'Description: {desc}')

        lines.append(f'\nPredicting for: {target_date}')
        if record.history:
            lines.append('Recent price history:')
            for day in record.history:
                lines.append(f"  {unix_to_date(day['t'])}: {day['p']:.3f}")

        lines.append('\nDoes this news have a causal relationship with the price change of this prediction market? '
                     'Rate higher only if the news could directly cause the market price to move. News that is merely '
                     'topically related but would not causally drive a price change should receive a low score.')
        return '\n'.join(lines)

    def _build_no_news_prompt(self, record: Record, target: Dict[str, float]) -> str:
        lines = [f'Event: {record.question}']
        if record.description:
            lines.append(f'Description: {record.description}')

        lines.append('\nRecent price history:')
        for day in record.history:
            lines.append(f"  {unix_to_date(day['t'])}: {day['p']:.3f}")

        target_date = unix_to_date(target['t'])
        lines.append('\nNews: No relevant news.')
        lines.append(f'\nPredict the probability on {target_date}:')
        return '\n'.join(lines)
