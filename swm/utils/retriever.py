# retriever.py

from pathlib import Path
from typing import List, Dict, Optional

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..data import PolyMarketData


class Retriever:
    def __init__(
        self,
        retriever_name: str,
        cache_dir: str,
        max_seq_length: int = 512,
        top_k: int = 50,
        retriever_batch_size: int = 32,
    ):
        self.retriever_name = retriever_name
        self.cache_dir = Path(cache_dir)
        self.max_seq_length = max_seq_length
        self.top_k = top_k
        self.retriever_batch_size = retriever_batch_size
        self.sentence_transformer = SentenceTransformer(
            retriever_name, device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.index = None
        self.corpus = {}
        self.corpus_ids = []
        self.embeddings = None
        self.market_embeddings = {}

    def setup_corpus(self, corpus_markets: List[PolyMarketData]) -> None:
        self.corpus_ids = []
        embeddings_list = []
        for i in range(0, len(corpus_markets), self.retriever_batch_size):
            batch = corpus_markets[i : i + self.retriever_batch_size]
            queries = [
                f"{m.question} {m.description or ''}"[: self.max_seq_length]
                for m in batch
            ]
            batch_embeddings = self.sentence_transformer.encode(
                queries, batch_size=self.retriever_batch_size, show_progress_bar=False
            )
            for j, market in enumerate(batch):
                self.corpus_ids.append(market.market_id)
                self.market_embeddings[market.market_id] = batch_embeddings[j]
                embeddings_list.append(batch_embeddings[j])
        self.embeddings = np.vstack(embeddings_list)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        self.corpus = {m.market_id: m for m in corpus_markets}

    def find_similar(
        self, market: PolyMarketData, k: Optional[int] = None
    ) -> List[PolyMarketData]:
        k = k or self.top_k
        if market.market_id not in self.market_embeddings:
            embedding = self._compute_embedding(market)
            self.market_embeddings[market.market_id] = embedding
            new_embedding = embedding.reshape(1, -1)
            self.index.add(new_embedding)
            self.embeddings = np.vstack([self.embeddings, new_embedding])
            self.corpus_ids.append(market.market_id)
            self.corpus[market.market_id] = market
        query_embedding = self.market_embeddings[market.market_id].reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        return [self.corpus[self.corpus_ids[idx]] for idx in indices[0]]

    def _compute_embedding(self, market: PolyMarketData) -> np.ndarray:
        query = f"{market.question} {market.description or ''}"[: self.max_seq_length]
        return self.sentence_transformer.encode([query])[0]
