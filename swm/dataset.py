"""Dataset classes for v6 records.

One v6 record == one training example. See data/v6/README.md for schema.
"""

import random
from typing import Any, Dict, List, Optional

import torch
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
            p,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors='pt',
        )
        ids_list.append(enc['input_ids'].squeeze(0))
        mask_list.append(enc['attention_mask'].squeeze(0))
    return {
        'input_ids': pad_sequence(
            ids_list,
            batch_first=True,
            padding_value=tokenizer.pad_token_id or 0,
        ),
        'attention_mask': pad_sequence(mask_list, batch_first=True, padding_value=0),
    }


def collate_padded_groups(
    batch: List[Dict[str, Any]],
    pad_token_id: int,
    max_seq_length: int,
) -> Dict[str, Any]:
    """Right-pad each item's (N_i, L_i) prompts to a common length and flatten.

    Shared by the world_model and attributer collators: both turn a batch of
    per-record prompt groups into one (sum N_i, L) tensor plus a group_ids
    vector and the carried market/event/t metadata. Callers add their own
    per-item tensors (labels/weights vs p_dist).
    """
    max_len = min(max(item['input_ids'].size(-1) for item in batch), max_seq_length)
    input_ids, attention_mask, group_ids = [], [], []
    market_ids, event_ids, ts = [], [], []
    for group_idx, item in enumerate(batch):
        ids = item['input_ids'][:, :max_len]
        padded = torch.nn.functional.pad(
            ids,
            (0, max_len - ids.size(-1)),
            value=pad_token_id,
        )
        input_ids.append(padded)
        attention_mask.append((padded != pad_token_id).long())
        group_ids.append(torch.full((padded.size(0),), group_idx, dtype=torch.long))
        market_ids.append(item['market_id'])
        event_ids.append(item['event_id'])
        ts.append(item['t'])
    return {
        'input_ids': torch.cat(input_ids, dim=0),
        'attention_mask': torch.cat(attention_mask, dim=0),
        'group_ids': torch.cat(group_ids, dim=0),
        'market_ids': market_ids,
        'event_ids': event_ids,
        'ts': ts,
    }


def build_attributer_news_prompt(
    record: Record,
    target: Dict[str, float],
    news: Dict[str, Any],
) -> str:
    """Per-news causal-attribution prompt. Single source of truth for both the
    attributer training dataset and real-time inference (attribute_record)."""
    # News-first ordering: protect the discriminative tail from right-truncation.
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

    lines.append(f"\nPredicting for: {unix_to_date(target['t'])}")
    if record.history:
        lines.append('Recent price history:')
        for day in record.history:
            lines.append(f"  {unix_to_date(day['t'])}: {day['p']:.3f}")

    lines.append(
        '\nDoes this news have a causal relationship with the price change of this prediction market? '
        'Rate higher only if the news could directly cause the market price to move. News that is merely '
        'topically related but would not causally drive a price change should receive a low score.'
    )
    return '\n'.join(lines)


def build_attributer_no_news_prompt(record: Record, target: Dict[str, float]) -> str:
    """The "no relevant news" option, shared by attributer training + inference."""
    lines = [f'Event: {record.question}']
    if record.description:
        lines.append(f'Description: {record.description}')

    lines.append('\nRecent price history:')
    for day in record.history:
        lines.append(f"  {unix_to_date(day['t'])}: {day['p']:.3f}")

    lines.append('\nNews: No relevant news.')
    lines.append(f"\nPredict the probability on {unix_to_date(target['t'])}:")
    return '\n'.join(lines)


class MultiEventWorldModelDataset(Dataset):
    """WorldModel training set: predict target.p from (question, history, news).

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
        null_subsample_ratio: float = 1.0,
        predict_delta: bool = True,
        include_nonews_candidate: bool = False,
        null_rho0: float = 1.0,
        odds_eps: float = 1e-3,
        odds_temp: float = 1.0,
        direct_soft_routing: bool = False,
    ):
        self.include_nonews_candidate = include_nonews_candidate
        self.direct_soft_routing = direct_soft_routing
        self.null_rho0 = null_rho0
        self.odds_eps = odds_eps
        self.odds_temp = odds_temp
        super().__init__()
        self.records = records
        self.tokenizer = tokenizer
        self.max_news = max_news
        self.max_seq_length = max_seq_length
        self.null_subsample_ratio = null_subsample_ratio
        self.predict_delta = predict_delta
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

        for record in tqdm(self.records, desc='Building world_model datapoints'):
            if not record.target or not record.news:
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
            raw = torch.tensor([w for _, w in items], dtype=torch.float).clamp(min=0.0)
            if getattr(self, 'direct_soft_routing', False):
                # PRIOR attributer (odds-trained) already emits the categorical
                # routing weights pi_i directly (per-record sum < 1; the missing
                # 1-sum is the implicit no-news mass). Use them AS-IS: clamp to
                # [0,1], mask any None candidate to 0, NO odds transform, NO
                # renorm. The no-news remainder -> 0 shrinks weak/null events.
                is_news = torch.tensor(
                    [news is not None for news, _ in items], dtype=torch.float
                )
                weights = raw.clamp(0.0, 1.0) * is_news
            else:
                # Independent per-news Bernoulli scores a_i -> joint (k+1) categorical
                # over (news, no-news) via odds + a null prior mass rho0:
                #   o_i = ((a_i+eps)/(1-a_i+eps))^(1/T),  o_0 = rho0
                #   pi_i = o_i / (rho0 + sum_j o_j)   (news weights sum to 1-pi_0)
                # The null mass pi_0 = rho0/(rho0+sum o) is NOT a candidate here:
                # no-news prediction is fixed 0, so news weights summing to <1
                # shrink the aggregate for low-confidence events. Loss/predict
                # must NOT renormalize these to 1.
                # A no-news candidate (news is None, e.g. a null event's only
                # candidate) predicts 0 and contributes 0: mask its odds to 0 so
                # null events aggregate to 0 (route-to-0), not the untrained
                # no-news-prompt output. Has-news events have no None candidate.
                is_news = torch.tensor(
                    [news is not None for news, _ in items], dtype=torch.float
                )
                a = raw.clamp(0.0, 1.0 - 1e-6)
                o = (a + self.odds_eps) / (1.0 - a + self.odds_eps) * is_news
                if self.odds_temp != 1.0:
                    o = o.pow(1.0 / self.odds_temp)
                weights = o / (self.null_rho0 + o.sum())

            before_p = (
                float(record.history[-1].get('p', 0.5))
                if record.history
                else float(target.get('p', 0.5))
            )
            target_p = float(target.get('p', 0.5))
            label_val = target_p - before_p if self.predict_delta else target_p
            datapoints.append(
                {
                    **_pack_prompts(self.tokenizer, prompts, self.max_seq_length),
                    'weights': weights,
                    'label': torch.tensor(label_val, dtype=torch.float),
                    'before_price': torch.tensor(before_p, dtype=torch.float),
                    'market_id': record.market_id,
                    'event_id': record.event_id,
                    't': target.get('t'),
                }
            )

        if self.null_subsample_ratio < 1.0:
            tot = null_kept + has_kept
            print(
                f'[MultiEventWorldModelDataset] null_subsample_ratio={self.null_subsample_ratio} '
                f'→ has_news={has_kept} ({100 * has_kept / max(tot, 1):.1f}%)  '
                f'null_kept={null_kept} ({100 * null_kept / max(tot, 1):.1f}%)  '
                f'null_dropped={null_dropped}'
            )

        return datapoints

    def _select_items(
        self,
        record: Record,
        rng: random.Random,
    ) -> Optional[List]:
        """List of (news_or_None, weight) tuples for this record, or None to drop."""
        positives = self._top_positives(record)
        if positives:
            # No-routing inference: the attributer's softmax is over (news ∪
            # {no-news}), so the news scores sum to <1 and the residual mass is
            # the no-news weight. Inject it as a regular weighted candidate so
            # the world_model blends news predictions with its no-news prediction
            # — no explicit null-gate / routing needed. Inference-only.
            if getattr(self, 'include_nonews_candidate', False):
                residual = 1.0 - sum(float(w) for _, w in positives)
                if residual > 1e-6:
                    positives = positives + [(None, residual)]
            return positives
        if rng.random() >= self.null_subsample_ratio:
            return None
        return [(None, 1.0)]

    def _top_positives(self, record: Record) -> List:
        n = len(record.news)
        # posterior (oracle) positive scores, idx -> score
        post = {
            a['news_idx']: float(a['score'])
            for a in record.attributions
            if 0 <= a.get('news_idx', -1) < n and float(a.get('score') or 0) > 0
        }
        items = [(record.news[i], s) for i, s in post.items()]
        items.sort(key=lambda p: -p[1])
        return items[: self.max_news]

    def _build_prompt(
        self,
        record: Record,
        target: Dict[str, float],
        news: Optional[Dict[str, Any]],
    ) -> str:
        # News-first ordering: protect the news signal from right-truncation at
        # max_seq_length. A long price history would otherwise push the news and
        # the trailing "Predict ...:" anchor (the last-token pooling target) past
        # the cutoff, leaving the model to extrapolate from history alone.
        target_date = unix_to_date(target['t'])
        if news is None:
            lines = ['News: No relevant news.']
        else:
            lines = [f'News: {news.get("title", "")}']
            if news.get('description'):
                lines.append(news['description'])

        lines.append(f'\nEvent: {record.question}')
        if record.description:
            lines.append(f'Description: {record.description}')

        lines.append('\nRecent price history:')
        for day in record.history:
            lines.append(f"  {unix_to_date(day['t'])}: {day['p']:.3f}")

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
        null_odds: float = 1.0,
        odds_eps: float = 1e-3,
        odds_temp: float = 1.0,
    ):
        super().__init__()
        self.records = records
        self.tokenizer = tokenizer
        self.max_news = max_news
        self.max_seq_length = max_seq_length
        self.include_negatives = include_negatives
        self.null_subsample_ratio = null_subsample_ratio
        # Target distribution over (news ∪ {no-news}): map each posterior score
        # a_i to odds o_i=((a_i+eps)/(1-a_i+eps))**(1/T), give the null slot a
        # fixed raw mass rho_0=null_odds, then normalize. Weak scores -> high
        # no-news (e.g. [0.2,0,0]->[no-news .8, .2,0,0]); null records -> no-news≈1
        # emerge naturally; no-news mass is nonzero for has-news so the forward-KL
        # gradient suppresses it.
        self.null_odds = null_odds
        self.odds_eps = odds_eps
        self.odds_temp = odds_temp
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

            use_idxs = self._select_news_idxs(record, scores_by_idx, rng)
            prompts = [
                build_attributer_news_prompt(record, target, record.news[i])
                for i in use_idxs
            ]
            prompts.append(build_attributer_no_news_prompt(record, target))

            # odds construction: news o_i = ((a_i+eps)/(1-a_i+eps))**(1/T),
            # null slot raw mass = null_odds; normalize over (news ∪ {no-news}).
            a = torch.tensor(
                [scores_by_idx.get(i, 0.0) for i in use_idxs],
                dtype=torch.float,
            ).clamp(0.0, 1.0)
            odds = (a + self.odds_eps) / (1.0 - a + self.odds_eps)
            if self.odds_temp != 1.0:
                odds = odds.pow(1.0 / self.odds_temp)
            raw = torch.cat([odds, torch.tensor([self.null_odds], dtype=torch.float)])
            p_dist = raw / raw.sum()

            datapoints.append(
                {
                    **_pack_prompts(self.tokenizer, prompts, self.max_seq_length),
                    'p_dist': p_dist,
                    'market_id': record.market_id,
                    'event_id': record.event_id,
                    't': target.get('t'),
                }
            )

        if self.null_subsample_ratio < 1.0:
            tot = null_kept + has_kept
            print(
                f'[PriorAttributerDataset] null_subsample_ratio={self.null_subsample_ratio} '
                f'→ has_news={has_kept} ({100 * has_kept / max(tot, 1):.1f}%)  '
                f'null_kept={null_kept} ({100 * null_kept / max(tot, 1):.1f}%)  '
                f'null_dropped={null_dropped}'
            )
        return datapoints

    def _positive_scores(self, record: Record) -> Dict[int, float]:
        n = len(record.news)
        return {
            a['news_idx']: float(a['score'])
            for a in record.attributions
            if 0 <= a.get('news_idx', -1) < n and float(a.get('score') or 0) > 0
        }

    def _select_news_idxs(
        self,
        record: Record,
        scores_by_idx: Dict[int, float],
        rng: random.Random,
    ) -> List[int]:
        n = len(record.news)
        if not scores_by_idx:
            return list(range(min(n, self.max_news)))
        pos_idxs = sorted(scores_by_idx, key=lambda i: -scores_by_idx[i])
        if not self.include_negatives:
            return pos_idxs[: self.max_news]
        neg_idxs = [i for i in range(n) if i not in scores_by_idx]
        if not neg_idxs:
            return pos_idxs[: self.max_news]
        n_neg = min(len(neg_idxs), max(1, self.max_news // 2))
        n_pos = self.max_news - n_neg
        # Randomly sample negatives rather than always taking the first by
        # original order, which kept feeding the same easy/early negatives and
        # never surfaced the harder ones later in the list.
        neg_sample = rng.sample(neg_idxs, n_neg)
        return pos_idxs[:n_pos] + neg_sample
