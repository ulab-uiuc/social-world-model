import json
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_squared_error
from transformers import Trainer, TrainingArguments

from .data import PolyMarketData
from .dataset import PolyMarketDataset
from .utils.regressor import LLMRegressor


class RAGSocialWM:
    def __init__(
        self,
        model_name: str,
        retriever_name: str,
        cache_dir: str,
        corpus_markets: Optional[List[PolyMarketData]] = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: List[str] = None,
        epochs: int = 3,
        train_batch_size: int = 4,
        eval_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        logging_steps: int = 10,
        save_steps: int = 50,
        eval_steps: int = 50,
        fp16: bool = False,
        output_dir: str = './market_predictor',
        top_k: int = 5,
        retriever_batch_size: int = 32,
        max_seq_length: int = 512,
    ):
        self.model_name = model_name
        self.retriever_name = retriever_name
        self.cache_dir = Path(cache_dir)

        # LoRA parameters
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ['q_proj', 'v_proj']

        # Training parameters
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.fp16 = fp16
        self.output_dir = output_dir

        # Retriever parameters
        self.top_k = top_k
        self.retriever_batch_size = retriever_batch_size
        self.max_seq_length = max_seq_length

        self._initialize_components()

        if corpus_markets:
            self.setup_retriever(corpus_markets)

    def _initialize_components(self):
        self.model = None
        self.sentence_transformer = SentenceTransformer(
            self.retriever_name, device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.index = None
        self.corpus = None
        self.corpus_ids = None
        self.embeddings = None
        self.market_embeddings = {}

    def _get_lora_config(self) -> LoraConfig:
        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias='none',
            task_type='CAUSAL_LM',
            target_modules=self.target_modules,
        )

    def _get_trainer_args(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            max_grad_norm=self.max_grad_norm,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            eval_strategy='steps',
            save_strategy='steps',
            fp16=self.fp16,
            load_best_model_at_end=True,
            metric_for_best_model='loss',
            save_safetensors=False,
        )

    def _compute_embedding(self, market: PolyMarketData) -> np.ndarray:
        query = f"{market.question} {market.description or ''}"
        query = query[: self.max_seq_length]
        return self.sentence_transformer.encode([query])[0]

    def _compute_batch_embeddings(self, markets: List[PolyMarketData]) -> np.ndarray:
        queries = [
            f"{market.question} {market.description or ''}"[: self.max_seq_length]
            for market in markets
        ]
        return self.sentence_transformer.encode(
            queries, batch_size=self.retriever_batch_size, show_progress_bar=False
        )

    def setup_model(self) -> None:
        self.model = LLMRegressor(
            model_name=self.model_name, max_length=self.max_seq_length
        )
        self.model.llm = get_peft_model(self.model.llm, self._get_lora_config())

    def setup_retriever(self, corpus_markets: List[PolyMarketData]) -> None:
        self.cache_dir.mkdir(exist_ok=True)
        self._setup_corpus(corpus_markets)
        self._save_retriever_state()

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

    def _save_retriever_state(self) -> None:
        faiss.write_index(self.index, str(self.cache_dir / 'corpus_index.faiss'))
        np.save(str(self.cache_dir / 'corpus_embeddings.npy'), self.embeddings)
        with open(self.cache_dir / 'corpus_ids.json', 'w') as f:
            json.dump(self.corpus_ids, f)

    def find_similar(
        self, market: PolyMarketData, k: Optional[int] = None
    ) -> List[PolyMarketData]:
        k = k or self.top_k
        if market.market_id not in self.market_embeddings:
            self.market_embeddings[market.market_id] = self._compute_embedding(market)

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
                        value=self.model.tokenizer.pad_token_id,
                    )
                    for x in batch
                ]
            )
            attention_mask = torch.stack(
                [
                    torch.nn.functional.pad(
                        x['attention_mask'][:max_len],
                        (0, max_len - min(x['attention_mask'].size(0), max_len)),
                        value=0,
                    )
                    for x in batch
                ]
            )
            labels = torch.stack([x['labels'] for x in batch])
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
            }

        return collate_fn

    def train(
        self,
        train_data: List[PolyMarketData],
        valid_data: List[PolyMarketData],
    ) -> str:
        if self.model is None:
            self.setup_model()

        train_similar = {m.market_id: self.find_similar(m) for m in train_data}
        valid_similar = {m.market_id: self.find_similar(m) for m in valid_data}

        train_dataset = PolyMarketDataset(train_data, train_similar, self.model.tokenizer)
        valid_dataset = PolyMarketDataset(valid_data, valid_similar, self.model.tokenizer)

        best_model_dir = Path(self.output_dir) / 'checkpoint-best'

        trainer = Trainer(
            model=self.model,
            args=self._get_trainer_args(),
            data_collator=self._create_collate_fn(),
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=lambda eval_pred: {
                'mse': mean_squared_error(
                    eval_pred.label_ids,
                    eval_pred.predictions.squeeze()
                )
            },
        )

        trainer.train()

        trainer.save_model(best_model_dir)

    def predict(self, market: PolyMarketData) -> Dict[str, float]:
        similar_markets = self.find_similar(market)
        dataset = PolyMarketDataset(
            [market], {market.market_id: similar_markets}, self.model.tokenizer
        )

        if not dataset:
            return {}

        predictions = []
        labels = []
        with torch.no_grad():
            for i in range(len(dataset)):
                datapoint = dataset[i]
                inputs = {k: v.unsqueeze(0) for k, v in datapoint.items()}
                pred = self.model(**inputs)
                predictions.append(pred['predictions'][0].item())
                labels.append(datapoint['labels'].item())

        return predictions, labels

    def predict_batch(
        self, markets: List[PolyMarketData], batch_size: int = 8
    ) -> Dict[str, Dict[str, float]]:
        predictions = {}
        for i in range(0, len(markets), batch_size):
            batch = markets[i : i + batch_size]
            similar_markets = {m.market_id: self.find_similar(m) for m in batch}
            dataset = PolyMarketDataset(batch, similar_markets, self.model.tokenizer)

            if not dataset:
                continue

            with torch.no_grad():
                for i in range(len(dataset)):
                    datapoint = dataset[i]
                    inputs = {k: v.unsqueeze(0) for k, v in datapoint.items()}
                    pred = self.model(**inputs)
                    market_id = dataset.datapoints[i]['market_id']
                    outcome = dataset.datapoints[i]['outcome']

                    if market_id not in predictions:
                        predictions[market_id] = {}
                    predictions[market_id][outcome] = pred.item()

        return predictions

    def save(self, path: str) -> None:
        path = Path(path)
        path.mkdir(exist_ok=True)

        if self.model:
            self.model.save_pretrained(path / 'model', safe_serialization=False)

        retriever_info = {
            'corpus_ids': self.corpus_ids,
            'market_embeddings': {
                k: v.tolist() for k, v in self.market_embeddings.items()
            },
        }

        with open(path / 'retriever_info.json', 'w') as f:
            json.dump(retriever_info, f)
        faiss.write_index(self.index, str(path / 'index.faiss'))
        np.save(str(path / 'embeddings.npy'), self.embeddings)

    def load(self, path: str) -> 'RAGSocialWM':
        path = Path(path)

        if (path / 'model').exists():
            self.model = LLMRegressor.from_pretrained(
                path / 'model', max_length=self.max_seq_length
            )

        with open(path / 'retriever_info.json', 'r') as f:
            retriever_info = json.dump(f)
            self.corpus_ids = retriever_info['corpus_ids']
            self.market_embeddings = {
                k: np.array(v) for k, v in retriever_info['market_embeddings'].items()
            }

        self.index = faiss.read_index(str(path / 'index.faiss'))
        self.embeddings = np.load(str(path / 'embeddings.npy'))
        return self
