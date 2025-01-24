# swm.py

from pathlib import Path
from typing import Dict, List, Optional

import torch
from peft import LoraConfig
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments

from .data import PolyMarketData
from .dataset import PolyMarketDataset
from .utils.regressor import LLMRegressor, LLMRegressorConfig
from .utils.retriever import Retriever


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
        self.cache_dir = Path(cache_dir)
        self.retriever_batch_size = retriever_batch_size
        self.retriever = Retriever(
            retriever_name=retriever_name,
            cache_dir=cache_dir,
            max_seq_length=max_seq_length,
            top_k=top_k,
            retriever_batch_size=retriever_batch_size,
        )
        if corpus_markets:
            self.setup_retriever(corpus_markets)

    def setup_model(self) -> None:
        config = LLMRegressorConfig(
            base_model_name_or_path=self.model_name, max_length=self.max_seq_length
        )
        self.model = LLMRegressor(config, lora_config=self.lora_config)

    def setup_retriever(self, corpus_markets: List[PolyMarketData]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.retriever.setup_corpus(corpus_markets)

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
            if 'market_id' in batch[0] and 'outcome' in batch[0]:
                market_ids = [x['market_id'] for x in batch]
                outcomes = [x['outcome'] for x in batch]
                return {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels,
                    'market_ids': market_ids,
                    'outcomes': outcomes,
                }
            else:
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
        train_dataset = PolyMarketDataset(
            train_data, train_similar, self.tokenizer, self.cache_dir
        )
        valid_dataset = PolyMarketDataset(
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
    ) -> Dict[str, Dict[str, float]]:
        similar_markets = {m.market_id: self.retriever.find_similar(m) for m in markets}
        dataset = PolyMarketDataset(
            markets, similar_markets, self.tokenizer, self.cache_dir
        )
        results = {}
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._create_collate_fn(),
        )
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Predicting Batches'):
                input_ids = batch['input_ids'].to(self.model.llm.device)
                attention_mask = batch['attention_mask'].to(self.model.llm.device)
                labels = batch['labels'].to(self.model.llm.device)
                preds = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                pred_values = preds['predictions'].squeeze(-1).cpu().numpy()
                label_values = labels.squeeze(-1).cpu().numpy()
                for i, market_id in enumerate(batch['market_ids']):
                    outcome = batch['outcomes'][i]
                    pred_value = pred_values[i].item()
                    label_value = label_values[i].item()
                    if market_id not in results:
                        results[market_id] = {}
                    results[market_id][outcome] = {
                        'pred': pred_value,
                        'label': label_value,
                    }
        return results

    def save(self, path: str) -> None:
        if self.model:
            self.model.save_pretrained(path)

    def load(self, path: str) -> None:
        self.model = LLMRegressor.from_pretrained(path)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
