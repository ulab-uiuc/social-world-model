from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import torch
from peft import LoraConfig
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments

from .data import PolyMarketData
from .dataset import PolyMarketDataset
from .utils.regressor import LLMRegressor, LLMRegressorConfig


class RAGSocialWM:
    def __init__(
        self,
        model_name: str,
        retriever_name: str,
        cache_dir: str,
        lora_config: Optional[LoraConfig] = None,
        corpus_markets: Optional[List[PolyMarketData]] = None,
        max_seq_length: int = 512,
        top_k: int = 50,
        retriever_batch_size: int = 32,
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_length = max_seq_length
        self.top_k = top_k
        self.model = None
        self.lora_config = lora_config

        # Initialize the SentenceTransformer-based retriever
        self.sentence_transformer = SentenceTransformer(
            retriever_name, device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.cache_dir = Path(cache_dir)
        self.retriever_batch_size = retriever_batch_size

        self.index = None
        self.corpus = None
        self.corpus_ids = []
        self.embeddings = None
        self.market_embeddings = {}

        if corpus_markets:
            self.setup_retriever(corpus_markets)

    def setup_model(self, device: torch.device = torch.device('cpu')) -> None:
        """
        Initializes the LLMRegressor model with the given device.
        """
        config = LLMRegressorConfig(
            base_model_name_or_path=self.model_name, max_length=self.max_seq_length
        )
        self.model = LLMRegressor(config, lora_config=self.lora_config, device=device)

    def _compute_embedding(self, market: PolyMarketData) -> np.ndarray:
        query = f"{market.question} {market.description or ''}"
        query = query[: self.max_seq_length]
        return self.sentence_transformer.encode([query])[0]

    def _compute_batch_embeddings(self, markets: List[PolyMarketData]) -> np.ndarray:
        queries = [
            f"{m.question} {m.description or ''}"[: self.max_seq_length]
            for m in markets
        ]
        return self.sentence_transformer.encode(
            queries, batch_size=self.retriever_batch_size, show_progress_bar=False
        )

    def setup_retriever(self, corpus_markets: List[PolyMarketData]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._setup_corpus(corpus_markets)

    def _setup_corpus(self, corpus_markets: List[PolyMarketData]) -> None:
        self.corpus_ids = []
        embeddings_list = []

        for i in range(0, len(corpus_markets), self.retriever_batch_size):
            batch = corpus_markets[i : i + self.retriever_batch_size]
            batch_embeddings = self._compute_batch_embeddings(batch)
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
            # Add to FAISS index
            new_embedding = embedding.reshape(1, -1)
            self.index.add(new_embedding)
            self.embeddings = np.vstack([self.embeddings, new_embedding])
            self.corpus_ids.append(market.market_id)
            self.corpus[market.market_id] = market
        query_embedding = self.market_embeddings[market.market_id].reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        return [self.corpus[self.corpus_ids[idx]] for idx in indices[0]]

    def _create_collate_fn(self):
        def collate_fn(batch):
            max_len = max(x['input_ids'].size(0) for x in batch)
            max_len = min(max_len, self.max_seq_length)
            input_ids = torch.stack(
                [
                    torch.nn.functional.pad(
                        x['input_ids'][:max_len],
                        (0, max_len - min(x['input_ids'].size(0), max_len)),
                        value=self.tokenizer.pad_token_id,
                    )
                    for x in batch
                ]
            )
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            labels = torch.stack([x['labels'] for x in batch])
            market_ids = [x['market_id'] for x in batch]
            outcomes = [x['outcome'] for x in batch]
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'market_ids': market_ids,
                'outcomes': outcomes,
            }

        return collate_fn

    def train(
        self,
        train_data: List[PolyMarketData],
        valid_data: List[PolyMarketData],
        training_args: TrainingArguments,
        device: torch.device = torch.device('cpu'),
    ) -> str:
        if self.model is None:
            self.setup_model(device=device)

        train_similar = {m.market_id: self.find_similar(m) for m in train_data}
        valid_similar = {m.market_id: self.find_similar(m) for m in valid_data}

        train_dataset = PolyMarketDataset(
            train_data, train_similar, self.tokenizer, self.cache_dir
        )
        valid_dataset = PolyMarketDataset(
            valid_data, valid_similar, self.tokenizer, self.cache_dir
        )

        best_model_dir = Path(self.output_dir) / 'checkpoint-best'

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=self._create_collate_fn(),
            compute_metrics=lambda p: {
                'mse': mean_squared_error(p.label_ids, p.predictions)
            },
        )

        trainer.train()
        best_model_dir = Path(training_args.output_dir) / 'checkpoint-best'
        trainer.save_model(best_model_dir)
        return str(best_model_dir)

    def predict(
        self, markets: List[PolyMarketData], batch_size: int = 8
    ) -> Dict[str, Dict[str, float]]:
        similar_markets = {m.market_id: self.find_similar(m) for m in markets}
        dataset = PolyMarketDataset(
            markets, similar_markets, self.model.llm.config.pad_token_id, self.cache_dir
        )

        predictions = {}
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._create_collate_fn(),
        )

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Predicting Batches'):
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                labels = batch['labels'].to(self.model.device)
                preds = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                pred_values = preds['predictions'].squeeze(-1).cpu().numpy()

                for i, market_id in enumerate(batch['market_ids']):
                    outcome = batch['outcomes'][i]
                    prediction_value = pred_values[i].item()
                    if market_id not in predictions:
                        predictions[market_id] = {}
                    predictions[market_id][outcome] = prediction_value
        return predictions

    def save(self, path: str) -> None:
        if self.model:
            self.model.save_pretrained(path)

    def load(self, path: str, device: torch.device = torch.device('cpu')) -> None:
        self.model = LLMRegressor.from_pretrained(path, device=device)
