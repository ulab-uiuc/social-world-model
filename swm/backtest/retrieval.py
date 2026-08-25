"""News <-> market relevance, standing in for the dataset's oracle attributions.

The `attributions` field in swmbench is a *posterior* label: it was produced
with the realised price move in hand, so feeding it to the model at backtest
time leaks the answer. This module scores relevance the way a live system
would have to -- bi-encoder cosine similarity between the market text and each
headline in the decision window -- and emits the top-k as attribution weights.

Scores are min-max free: raw cosine (roughly [0, 1] for a normalised
bi-encoder) is mapped through a temperature-softened rank so the downstream
odds transform in MultiEventWorldModelDataset sees a comparable scale to the
0.75/0.95 oracle scores.
"""

import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_MODEL = 'BAAI/bge-small-en-v1.5'
QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '


@dataclass
class RetrievedNews:
    news_index: int
    score: float
    similarity: float


@dataclass
class RetrievedMarket:
    market_id: str
    score: float
    similarity: float


class EmbeddingRetriever:
    """Bi-encoder retriever over a fixed news corpus and market corpus."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = 'cuda',
        batch_size: int = 256,
        cache_folder: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(
            model_name, device=device, cache_folder=cache_folder
        )
        self._news: np.ndarray | None = None
        self._markets: dict[str, np.ndarray] = {}

    def _encode(self, texts: Sequence[str], is_query: bool) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), 'f')
        prepared = list(texts)
        if is_query and 'bge' in self.model_name.lower():
            prepared = [QUERY_PREFIX + t for t in prepared]
        return self.model.encode(
            prepared,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(prepared) > 5000,
        ).astype('float32')

    def fit_news(self, texts: Sequence[str]) -> None:
        self._news = self._encode(texts, is_query=False)

    def fit_markets(self, market_ids: Sequence[str], texts: Sequence[str]) -> None:
        vectors = self._encode(texts, is_query=True)
        self._markets = dict(zip(market_ids, vectors))

    def similarities(self, market_id: str, news_indices: Sequence[int]) -> np.ndarray:
        if self._news is None or market_id not in self._markets:
            return np.zeros(len(news_indices), 'float32')
        if len(news_indices) == 0:
            return np.zeros(0, 'float32')
        return self._news[np.asarray(news_indices)] @ self._markets[market_id]

    def top_k(
        self,
        market_id: str,
        news_indices: Sequence[int],
        k: int,
        min_similarity: float = 0.0,
        temperature: float = 0.05,
        calibration: Optional['Calibration'] = None,
    ) -> list[RetrievedNews]:
        """Top-k headlines for a market, with attribution-shaped weights.

        The weights feed `MultiEventWorldModelDataset`'s odds transform, which
        reads per-news scores in [0, 1] and turns them into routing mass with a
        null slot: scores near 1 dominate, scores near 0 leave the mass on the
        implicit "no relevant news" option and shrink the prediction toward a
        zero delta. Preserving that behaviour is the whole point of scoring on
        *absolute* similarity (`calibration`) rather than a softmax over the
        retrieved set -- a pure softmax would hand the top hit ~0.95 even for a
        market nothing in the window is about, and the model would never route
        to null on the quiet cells the grid is full of.
        """
        if k <= 0 or len(news_indices) == 0:
            return []
        sims = self.similarities(market_id, news_indices)
        order = np.argsort(-sims)[:k]
        kept = [(int(news_indices[i]), float(sims[i])) for i in order]
        kept = [(idx, sim) for idx, sim in kept if sim >= min_similarity]
        if not kept:
            return []

        if calibration is not None:
            scores = [calibration(sim) for _, sim in kept]
        else:
            raw = np.array([sim for _, sim in kept], 'float64')
            weights = np.exp((raw - raw.max()) / max(temperature, 1e-6))
            weights = weights / weights.max()
            scores = [float(0.95 * w) for w in weights]

        # Zero-scored hits are kept, not dropped: `_top_positives` in the
        # world-model dataset filters on score > 0 itself, and keeping the full
        # top-k in the record makes it possible to see what retrieval surfaced
        # on a cell the model then routed to null.
        return [
            RetrievedNews(news_index=idx, score=float(score), similarity=sim)
            for (idx, sim), score in zip(kept, scores)
        ]

    def top_markets(
        self,
        news_index: int,
        market_ids: Sequence[str],
        k: int,
        min_similarity: float = 0.0,
        temperature: float = 0.05,
        calibration: Optional['Calibration'] = None,
    ) -> list[RetrievedMarket]:
        """Top-k markets for one headline -- the reverse of `top_k`.

        Used by the news-driven backtest: iterate the wire in time order and, for
        each headline, surface the markets it is most about. The same
        `calibration` is reused so an off-topic headline scores every market near
        zero and the builder can decline to trade -- exactly the null-routing the
        forward direction depends on. Scoring is on *absolute* similarity, not a
        softmax over the candidate markets, for the same reason.
        """
        if self._news is None or k <= 0 or len(market_ids) == 0:
            return []
        ids = [m for m in market_ids if m in self._markets]
        if not ids:
            return []
        news_vec = self._news[news_index]
        mat = np.stack([self._markets[m] for m in ids])
        sims = mat @ news_vec
        order = np.argsort(-sims)[:k]
        kept = [(ids[i], float(sims[i])) for i in order]
        kept = [(mid, sim) for mid, sim in kept if sim >= min_similarity]
        if not kept:
            return []

        if calibration is not None:
            scores = [calibration(sim) for _, sim in kept]
        else:
            raw = np.array([sim for _, sim in kept], 'float64')
            weights = np.exp((raw - raw.max()) / max(temperature, 1e-6))
            weights = weights / weights.max()
            scores = [float(0.95 * w) for w in weights]

        return [
            RetrievedMarket(market_id=mid, score=float(score), similarity=sim)
            for (mid, sim), score in zip(kept, scores)
        ]


@dataclass
class Calibration:
    """Maps a raw cosine similarity onto the oracle attributions' score range.

    `lo` is where attributed and unattributed news are indistinguishable and
    `hi` is where a hit looks as convincing as an oracle positive; both are
    estimated from TRAIN-split records only (see `calibrate`), so no test-period
    information reaches the mapping.
    """

    lo: float
    hi: float
    ceiling: float = 0.95

    def __call__(self, similarity: float) -> float:
        if self.hi <= self.lo:
            return 0.0
        frac = (similarity - self.lo) / (self.hi - self.lo)
        return float(min(max(frac, 0.0), 1.0) * self.ceiling)


def calibrate(
    positive_similarities: Sequence[float],
    negative_similarities: Sequence[float],
    lo_quantile: float = 0.95,
    hi_quantile: float = 0.95,
) -> Calibration:
    """Fit the similarity -> score mapping from labelled train-split pairs.

    `lo` sits at the top of the *negative* distribution: a headline has to beat
    95% of the unrelated news in its own window before it earns any weight at
    all. `hi` sits at the top of the *positive* distribution, where a hit is as
    convincing as a strong oracle attribution.

    The gap between the two is narrow -- on the train split a bi-encoder
    separates attributed from unattributed headlines at AUC ~0.74, not ~0.95 --
    so the quantiles are what stop every retrieved item from pinning at the
    ceiling. Left saturated, the model would receive a confident attribution on
    every cell and could never route to its null option, which is exactly the
    behaviour the quiet cells in grid mode depend on.
    """
    neg = sorted(negative_similarities)
    pos = sorted(positive_similarities)
    if not neg or not pos:
        return Calibration(lo=0.0, hi=1.0)
    lo = neg[min(len(neg) - 1, int(lo_quantile * len(neg)))]
    hi = pos[min(len(pos) - 1, int(hi_quantile * len(pos)))]
    if hi <= lo:
        hi = lo + 1e-3
    return Calibration(lo=float(lo), hi=float(hi))


def fit_calibration(
    train_records: Sequence[dict],
    retriever: EmbeddingRetriever,
    n_records: int,
    seed: int,
    verbose: bool = True,
) -> Calibration:
    """Similarity gap between oracle-positive and unattributed headlines.

    Fit on train records only. Both sides are measured inside the same record so
    the comparison is per-market, not across markets of differing verbosity.
    Shared by the grid and news-driven cell builders so they gate relevance on
    an identical, out-of-sample-safe mapping.
    """
    rng = random.Random(seed)
    sample = (
        list(train_records)
        if len(train_records) <= n_records
        else rng.sample(list(train_records), n_records)
    )
    market_ids, texts, seen_markets = [], [], set()
    for record in sample:
        mid = str(record['market_id'])
        if mid in seen_markets:
            continue
        seen_markets.add(mid)
        market_ids.append(mid)
        texts.append(
            f'{record.get("question", "")}\n{(record.get("description") or "")[:400]}'
        )
    retriever.fit_markets(market_ids, texts)

    news_texts, index_of = [], {}
    for record in sample:
        for item in record.get('news') or []:
            key = ((item.get('title') or '').strip(), item.get('published_at'))
            if key in index_of:
                continue
            index_of[key] = len(news_texts)
            body = (item.get('description') or '').strip()
            title = (item.get('title') or '').strip()
            news_texts.append(f'{title}\n{body}' if body != title else title)
    retriever.fit_news(news_texts)

    positives, negatives = [], []
    for record in sample:
        mid = str(record['market_id'])
        news = record.get('news') or []
        pos_idx = {
            a['news_idx']
            for a in record.get('attributions') or []
            if 0 <= a.get('news_idx', -1) < len(news) and float(a.get('score') or 0) > 0
        }
        if not pos_idx:
            continue
        corpus_idx, is_pos = [], []
        for i, item in enumerate(news):
            key = ((item.get('title') or '').strip(), item.get('published_at'))
            if key not in index_of:
                continue
            corpus_idx.append(index_of[key])
            is_pos.append(i in pos_idx)
        if not corpus_idx:
            continue
        sims = retriever.similarities(mid, corpus_idx)
        for sim, flag in zip(sims, is_pos):
            (positives if flag else negatives).append(float(sim))

    calibration = calibrate(positives, negatives)
    if verbose and positives and negatives:
        print(
            f'[calib] positives={len(positives)} (mean {statistics.fmean(positives):.3f}) '
            f'negatives={len(negatives)} (mean {statistics.fmean(negatives):.3f}) '
            f'-> lo={calibration.lo:.3f} hi={calibration.hi:.3f}'
        )
    return calibration
