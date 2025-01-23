import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
import torch
from peft import LLMRegression, LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_squared_error
from transformers import Trainer, TrainingArguments

from .data import PolyMarketData
from .dataset import PolyMarketDataset


class RAGSocialWM:
    def __init__(self, model_name: str = 'mistralai/Mistral-7B-v0.1'):
        self.model_name = model_name
        self.model = None
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.corpus = None
        self.corpus_ids = None
        self.embeddings = None

    def setup_model(self):
        self.model = LLMRegression(model_name=self.model_name)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias='none',
            task_type='CAUSAL_LM',
            target_modules=['q_proj', 'v_proj'],
        )
        self.model.llm = get_peft_model(self.model.llm, lora_config)
        return self.model

    def setup_retriever(
        self, corpus_markets: List[PolyMarketData], cache_dir: str = './market_cache'
    ):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)

        index_path = cache_dir / 'corpus_index.faiss'
        embeddings_path = cache_dir / 'corpus_embeddings.npy'
        corpus_ids_path = cache_dir / 'corpus_ids.json'

        if all(p.exists() for p in [index_path, embeddings_path, corpus_ids_path]):
            self.index = faiss.read_index(str(index_path))
            self.embeddings = np.load(str(embeddings_path))
            with open(corpus_ids_path, 'r') as f:
                self.corpus_ids = json.load(f)
        else:
            texts = []
            self.corpus_ids = []

            for market in corpus_markets:
                texts.append(f"{market.question} {market.discrption or ''}")
                self.corpus_ids.append(market.market_id)

            self.embeddings = self.sentence_transformer.encode(texts)
            self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
            self.index.add(self.embeddings)

            faiss.write_index(self.index, str(index_path))
            np.save(str(embeddings_path), self.embeddings)
            with open(corpus_ids_path, 'w') as f:
                json.dump(self.corpus_ids, f)

        self.corpus = {m.market_id: m for m in corpus_markets}

    def find_similar(self, market: PolyMarketData, k: int = 5) -> List[PolyMarketData]:
        query = f"{market.question} {market.discrption or ''}"
        query_embedding = self.sentence_transformer.encode([query])
        distances, indices = self.index.search(query_embedding, k)
        return [self.corpus[self.corpus_ids[idx]] for idx in indices[0]]

    def train(
        self,
        train_data: List[PolyMarketData],
        valid_data: List[PolyMarketData],
        corpus_markets: List[PolyMarketData],
        epochs: int = 3,
    ):
        if self.model is None:
            self.setup_model()
        if self.index is None:
            self.setup_retriever(corpus_markets)

        # Pre-compute similar markets for training data
        train_similar_markets = {
            market.market_id: self.find_similar(market) for market in train_data
        }
        valid_similar_markets = {
            market.market_id: self.find_similar(market) for market in valid_data
        }

        train_dataset = PolyMarketDataset(
            train_data, train_similar_markets, self.model.tokenizer
        )
        valid_dataset = PolyMarketDataset(
            valid_data, valid_similar_markets, self.model.tokenizer
        )

        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir='./market_predictor',
                num_train_epochs=epochs,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                learning_rate=1e-4,
                fp16=True,
                logging_steps=10,
                save_strategy='epoch',
                evaluation_strategy='steps',
                eval_steps=50,
                load_best_model_at_end=True,
                metric_for_best_model='loss',
            ),
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=lambda eval_pred: {
                'mse': mean_squared_error(
                    eval_pred.label_ids, eval_pred.predictions.squeeze()
                )
            },
        )
        trainer.train()
        return self

    def predict(
        self, test_data: List[PolyMarketData], corpus_markets: List[PolyMarketData]
    ) -> Dict[str, Dict[str, float]]:
        if self.index is None:
            self.setup_retriever(corpus_markets)

        test_similar_markets = {
            market.market_id: self.find_similar(market) for market in test_data
        }

        dataset = PolyMarketDataset(
            test_data, test_similar_markets, self.model.tokenizer
        )
        if len(dataset) == 0:
            return {}

        predictions = {}
        current_market_idx = 0
        current_market_id = test_data[current_market_idx].market_id
        market_predictions = {}

        for i in range(len(dataset)):
            datapoint = dataset[i]
            with torch.no_grad():
                pred = self.model(**{k: v.unsqueeze(0) for k, v in datapoint.items()})
                value = pred.item()

            if (
                i < len(dataset) - 1
                and dataset.datapoints[i]['market_id']
                != dataset.datapoints[i + 1]['market_id']
            ):
                predictions[current_market_id] = market_predictions
                current_market_idx += 1
                current_market_id = test_data[current_market_idx].market_id
                market_predictions = {}

            market_predictions[dataset.datapoints[i]['outcome']] = value

        if market_predictions:
            predictions[current_market_id] = market_predictions

        return predictions

    def save(self, path: str):
        if self.model is None:
            raise ValueError('No model to save')
        self.model.save_pretrained(path)

    def load(self, path: str):
        self.model = LLMRegression.from_pretrained(path)
        return self
