# swm.py

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from peft import LoraConfig
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments

from .data import PolyMarketData
from .dataset import BasicPolyMarketDataset, RAGPolyMarketDataset
from .utils.regressor import LLMRegressor, LLMRegressorConfig
from .utils.retriever import SimilarityBasedPolyMarketRetriever


class BasicSocialWM:
    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        max_seq_length: int = 512,
        lora_config: Optional[LoraConfig] = None,
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_length = max_seq_length
        self.model = None
        self.cache_dir = Path(cache_dir)
        self.lora_config = lora_config

    def setup_model(self) -> None:
        config = LLMRegressorConfig(
            base_model_name_or_path=self.model_name, max_length=self.max_seq_length
        )
        self.model = LLMRegressor(config, lora_config=self.lora_config)

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
            labels = torch.stack([x['label'] for x in batch])
            market_ids = [x['market_id'] for x in batch]
            event_ids = [x['event_id'] for x in batch]
            ts = [x['t'] for x in batch]
            outcomes = [x['outcome'] for x in batch]
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'market_ids': market_ids,
                'event_ids': event_ids,
                'ts': ts,
                'outcomes': outcomes,
            }

        return collate_fn

    def train(
        self,
        train_data: List[PolyMarketData],
        valid_data: List[PolyMarketData],
        training_args: TrainingArguments,
    ) -> str:
        if self.model is None:
            self.setup_model()

        train_dataset = BasicPolyMarketDataset(
            train_data, self.tokenizer, self.cache_dir
        )
        valid_dataset = BasicPolyMarketDataset(
            valid_data, self.tokenizer, self.cache_dir
        )

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
        dataset = BasicPolyMarketDataset(markets, self.tokenizer, self.cache_dir)
        results = {}
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._create_collate_fn(),
        )

        results = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Predicting Batches'):
                input_ids = batch['input_ids'].to(self.model.llm.device)
                attention_mask = batch['attention_mask'].to(self.model.llm.device)
                labels = batch['labels'].to(self.model.llm.device)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                predictions = outputs['predictions'].view(-1).cpu().numpy().tolist()

                for i in range(len(predictions)):
                    results.append(
                        {
                            'market_id': batch['market_ids'][i],
                            'event_id': batch['event_ids'][i],
                            't': batch['ts'][i],
                            'outcome': batch['outcomes'][i],
                            'prediction': predictions[i],
                            'ground_truth': labels[i].item(),
                        }
                    )
        return results

    def save(self, path: str) -> None:
        if self.model:
            self.model.save_pretrained(path)

    def load(self, path: str) -> None:
        self.model = LLMRegressor.from_pretrained(path)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')


class RAGSocialWM(BasicSocialWM):
    def __init__(
        self,
        model_name: str,
        retriever_name: str,
        cache_dir: str,
        lora_config: Optional[LoraConfig] = None,
        corpus_markets: Optional[List[PolyMarketData]] = None,
        max_seq_length: int = 512,
        retriever_top_k: int = 50,
        retriever_batch_size: int = 32,
    ):
        super().__init__(
            model_name=model_name,
            cache_dir=cache_dir,
            max_seq_length=max_seq_length,
            lora_config=lora_config,
        )
        self.retriever_name = retriever_name
        self.retriever_top_k = retriever_top_k
        self.retriever_batch_size = retriever_batch_size
        self.retriever = SimilarityBasedPolyMarketRetriever(
            retriever_name=retriever_name,
            cache_dir=cache_dir,
            max_seq_length=max_seq_length,
            retriever_top_k=retriever_top_k,
            retriever_batch_size=retriever_batch_size,
        )
        if corpus_markets:
            self.setup_retriever(corpus_markets)

    def setup_retriever(self, corpus_markets: List[PolyMarketData]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.retriever.setup_corpus(corpus_markets)

    def train(
        self,
        train_data: List[PolyMarketData],
        valid_data: List[PolyMarketData],
        training_args: TrainingArguments,
    ) -> str:
        if self.model is None:
            self.setup_model()

        train_similar = {
            m.market_id: self.retriever.find_similar(m) for m in train_data
        }
        valid_similar = {
            m.market_id: self.retriever.find_similar(m) for m in valid_data
        }
        train_dataset = RAGPolyMarketDataset(
            train_data, train_similar, self.tokenizer, self.cache_dir
        )
        valid_dataset = RAGPolyMarketDataset(
            valid_data, valid_similar, self.tokenizer, self.cache_dir
        )

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
    ) -> List[Dict[str, Union[str, float]]]:
        similar_markets = {m.market_id: self.retriever.find_similar(m) for m in markets}
        dataset = RAGPolyMarketDataset(
            markets, similar_markets, self.tokenizer, self.cache_dir
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._create_collate_fn(),
        )

        results = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Predicting Batches'):
                input_ids = batch['input_ids'].to(self.model.llm.device)
                attention_mask = batch['attention_mask'].to(self.model.llm.device)
                labels = batch['labels'].to(self.model.llm.device)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                predictions = outputs['predictions'].view(-1).cpu().numpy().tolist()

                for i in range(len(predictions)):
                    results.append(
                        {
                            'market_id': batch['market_ids'][i],
                            'event_id': batch['event_ids'][i],
                            't': batch['ts'][i],
                            'outcome': batch['outcomes'][i],
                            'prediction': predictions[i],
                            'ground_truth': labels[i].item(),
                        }
                    )

        return results

    def save(self, path: str) -> None:
        if self.model:
            self.model.save_pretrained(path)

    def load(self, path: str) -> None:
        self.model = LLMRegressor.from_pretrained(path)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')


"""
class BasicSocialWMWithEvent:
    def __init__(
        self,
        predictor: BasicPredictor,
        reasoner: BasicPriorReasoner,
    ):
        self.predictor = predictor
        self.reasoner = reasoner

    def analyze_and_predict(
        self,
        markets: List[PolyMarketData],
        news_data: List[DailyNewsData],
        date: str,
    ) -> Dict[str, Dict[str, Any]]:
        # Get market changes for reasoning
        market_changes = []
        for market in markets:
            if not market.daily_time_series or 'Yes' not in market.daily_time_series:
                continue
            series = market.daily_time_series['Yes']
            for i, point in enumerate(series):
                if datetime.fromtimestamp(point['t']).strftime('%Y-%m-%d') == date:
                    if i > 0:
                        change = {
                            'market': market,
                            'prev_point': series[i - 1],
                            'current_point': point,
                        }
                        market_changes.append(change)
                    break

        # Get news importance weights
        reasoning_results = self.reasoner.analyze(news_data, market_changes, date)

        # Use predictor with weighted news
        prediction_results = self.predictor.predict(markets, reasoning_results)

        # Combine results
        combined_results = {}
        for market_id, predictions in prediction_results.items():
            combined_results[market_id] = {
                'predictions': predictions,
                'news_weights': {
                    result['news'].id: result['score'] for result in reasoning_results
                },
            }

        return combined_results
"""
