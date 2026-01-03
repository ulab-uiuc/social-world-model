from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from peft import LoraConfig
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments

from .data import MarketData
from .dataset import MultiEventForecasterDataset, RAGMultiEventForecasterDataset
from .utils.regressor import LLMRegressor, LLMRegressorConfig
from .utils.retriever import SimilarityBasedMarketRetriever


class WeightedTrainer(Trainer):
    def _process_group(
        self, model, inputs, weights, group_indices, labels, is_prediction=False
    ):
        group_size = len(group_indices)
        chunk_size = 8
        acc_pred = 0

        group_weights = weights[group_indices]
        # Add epsilon to avoid division by zero
        normalized_weights = group_weights / (group_weights.sum() + 1e-8)

        for i in range(0, group_size, chunk_size):
            chunk_indices = group_indices[i : i + chunk_size]
            chunk_inputs = {
                'input_ids': inputs['input_ids'][chunk_indices],
                'attention_mask': inputs['attention_mask'][chunk_indices],
            }
            chunk_weights = normalized_weights[i : i + chunk_size]

            with (
                torch.amp.autocast('cuda', enabled=self.args.fp16)
                if not is_prediction
                else torch.no_grad()
            ):
                chunk_preds = model(**chunk_inputs)
                chunk_preds = chunk_preds.view(-1)
                acc_pred += (chunk_preds * chunk_weights).sum()

        group_label = labels[group_indices[0]]
        loss = torch.nn.functional.mse_loss(acc_pred, group_label)

        return loss, acc_pred, group_label

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop('labels')
        weights = inputs.pop('weights')
        group_ids = inputs.pop('group_ids')

        total_loss = 0
        num_valid_groups = 0
        for group in torch.unique(group_ids):
            group_indices = torch.where(group_ids == group)[0]
            loss, _, _ = self._process_group(
                model, inputs, weights, group_indices, labels
            )
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss
                num_valid_groups += 1

        return total_loss / max(num_valid_groups, 1)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.pop('labels')
        weights = inputs.pop('weights')
        group_ids = inputs.pop('group_ids')

        all_losses, all_preds, all_labels = [], [], []
        for group in torch.unique(group_ids):
            group_indices = torch.where(group_ids == group)[0]
            loss, pred, label = self._process_group(
                model, inputs, weights, group_indices, labels, is_prediction=True
            )

            if not torch.isnan(loss) and not torch.isinf(loss):
                all_losses.append(loss)
                all_preds.append(pred)
                all_labels.append(label)

        return (
            torch.stack(all_losses).mean(),
            torch.stack(all_preds),
            torch.stack(all_labels),
        )


class MultiEventForecaster:
    """
    Forecaster that uses multiple attributed events to predict market prices.
    
    Training: Uses precomputed attributions stored in market.attributions
    Inference: Uses PriorAttributer to generate attributions on-the-fly
    """
    def __init__(
        self,
        model_name: str,
        cache_dir: str,
        max_seq_length: int = 512,
        lora_config: Optional[LoraConfig] = None,
        gradient_checkpointing: bool = False,
        max_news_per_bp: int = 50,
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_length = max_seq_length
        self.model = None
        self.cache_dir = Path(cache_dir)
        self.lora_config = lora_config
        self.gradient_checkpointing = gradient_checkpointing
        self.max_news_per_bp = max_news_per_bp

    def setup_model(self) -> None:
        config = LLMRegressorConfig(
            base_model_name_or_path=self.model_name, max_length=self.max_seq_length
        )
        self.model = LLMRegressor(config, lora_config=self.lora_config)
        
        # Enable gradient checkpointing to save memory
        if self.gradient_checkpointing:
            if hasattr(self.model.llm, 'gradient_checkpointing_enable'):
                # use_reentrant=False is required for inputs without requires_grad
                self.model.llm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                print("Gradient checkpointing enabled (use_reentrant=False)")

    def _create_collate_fn(self):
        def collate_fn(batch):
            max_len = max(
                max(x['input_ids'].size(-1) for x in batch), self.max_seq_length
            )

            all_input_ids = []
            all_attention_masks = []
            all_labels = []
            all_weights = []
            all_group_ids = []
            all_market_ids = []
            all_event_ids = []
            all_ts = []

            for group_idx, item in enumerate(batch):
                padded_inputs = torch.nn.functional.pad(
                    item['input_ids'],
                    (0, max_len - item['input_ids'].size(-1)),
                    value=self.tokenizer.pad_token_id,
                )
                all_input_ids.append(padded_inputs)
                mask = (padded_inputs != self.tokenizer.pad_token_id).long()
                all_attention_masks.append(mask)
                all_labels.append(item['label'])
                all_weights.append(item['weights'])
                all_group_ids.append(torch.full((len(item['weights']),), group_idx))
                all_market_ids.append(item['market_id'])
                all_event_ids.append(item['event_id'])
                all_ts.append(item['t'])

            return {
                'input_ids': torch.cat(all_input_ids),
                'attention_mask': torch.cat(all_attention_masks),
                'labels': torch.stack(all_labels),  # labels are 0-D tensors, use stack
                'weights': torch.cat(all_weights),
                'group_ids': torch.cat(all_group_ids),
                'market_ids': all_market_ids,
                'event_ids': all_event_ids,
                'ts': all_ts,
            }

        return collate_fn

    def train(
        self,
        train_data: List[MarketData],
        valid_data: List[MarketData],
        training_args: TrainingArguments,
    ) -> str:
        """
        Train the forecaster using precomputed attributions.
        
        Args:
            train_data: List of markets with precomputed attributions in market.attributions
            valid_data: List of markets with precomputed attributions
            training_args: HuggingFace TrainingArguments
            
        Returns:
            Path to best model checkpoint
        """
        if self.model is None:
            self.setup_model()

        train_dataset = MultiEventForecasterDataset(
            markets=train_data,
            tokenizer=self.tokenizer,
            cache_dir=self.cache_dir,
            max_news_per_bp=self.max_news_per_bp,
        )
        valid_dataset = MultiEventForecasterDataset(
            markets=valid_data,
            tokenizer=self.tokenizer,
            cache_dir=self.cache_dir,
            max_news_per_bp=self.max_news_per_bp,
        )

        trainer = WeightedTrainer(
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
        self,
        markets: List[MarketData],
        attributer: Any = None,
        batch_size: int = 8,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Predict market prices.
        
        Args:
            markets: List of markets. Should have either:
                - Precomputed attributions in market.attributions, OR
                - Pass an attributer to generate attributions on-the-fly
            attributer: Optional PriorAttributer to generate attributions for inference
            batch_size: Batch size for prediction
            
        Returns:
            List of prediction results
        """
        # If attributer provided, generate attributions on-the-fly
        if attributer is not None:
            markets = self._generate_attributions(markets, attributer)
        
        dataset = MultiEventForecasterDataset(
            markets=markets,
            tokenizer=self.tokenizer,
            cache_dir=self.cache_dir,
            max_news_per_bp=self.max_news_per_bp,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._create_collate_fn(),
        )
        self.model.eval()
        results = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Predicting Batches'):
                input_ids = batch['input_ids'].to(self.model.llm.device)
                attention_mask = batch['attention_mask'].to(self.model.llm.device)
                labels = batch['labels'].to(self.model.llm.device)
                weights = batch['weights'].to(self.model.llm.device)
                group_ids = batch['group_ids'].to(self.model.llm.device)

                for group_idx in range(len(batch['event_ids'])):
                    group_mask = group_ids == group_idx
                    group_inputs = {
                        'input_ids': input_ids[group_mask],
                        'attention_mask': attention_mask[group_mask],
                    }

                    group_preds = self.model(**group_inputs).view(-1)

                    group_weights = weights[group_mask]
                    group_weights = group_weights / group_weights.sum()

                    weighted_pred = (group_preds * group_weights).sum()

                    results.append(
                        {
                            'event_id': batch['event_ids'][group_idx],
                            'market_id': batch['market_ids'][group_idx],
                            't': batch['ts'][group_idx],
                            'prediction': weighted_pred.item(),
                            'ground_truth': labels[group_mask][0].item(),
                        }
                    )
        return results

    def _generate_attributions(
        self, 
        markets: List[MarketData], 
        attributer: Any,
        window_size: int = 5,
    ) -> List[MarketData]:
        """Generate attributions for markets using the provided attributer."""
        for market in tqdm(markets, desc='Generating attributions'):
            if not market.daily_time_series or 'Yes' not in market.daily_time_series:
                continue
                
            series = market.daily_time_series['Yes']
            if len(series) <= window_size:
                continue
            
            attributions = {}
            for start_idx in range(len(series) - window_size):
                target = series[start_idx + window_size]
                target_ts = str(target['t'])
                
                # Call attributer to get events for this timestamp
                events = attributer.attribute(target['t'], market)
                if events:
                    attributions[target_ts] = [
                        {'news': e['news'], 'score': e['score']} 
                        for e in events
                    ]
            
            market.attributions = attributions
        
        return markets

    def save(self, path: str) -> None:
        if self.model:
            self.model.save_pretrained(path)

    def load(self, path: str) -> None:
        self.model = LLMRegressor.from_pretrained(path)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')


class RAGMultiEventForecaster(MultiEventForecaster):
    """
    RAG-enhanced MultiEventForecaster that retrieves similar markets for context.
    
    Combines:
    1. Attribution-based multi-event prediction
    2. RAG retrieval of similar historical markets
    """
    def __init__(
        self,
        model_name: str,
        retriever_name: str,
        cache_dir: str,
        corpus_markets: Optional[List[MarketData]] = None,
        max_seq_length: int = 512,
        lora_config: Optional[LoraConfig] = None,
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
        self.retriever = SimilarityBasedMarketRetriever(
            retriever_name=retriever_name,
            cache_dir=cache_dir,
            max_seq_length=max_seq_length,
            retriever_top_k=retriever_top_k,
            retriever_batch_size=retriever_batch_size,
        )
        if corpus_markets:
            self.setup_retriever(corpus_markets)

    def setup_retriever(self, corpus_markets: List[MarketData]) -> None:
        """Setup the retriever with a corpus of markets."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.retriever.setup_corpus(corpus_markets)

    def train(
        self,
        train_data: List[MarketData],
        valid_data: List[MarketData],
        training_args: TrainingArguments,
    ) -> str:
        """
        Train the RAG forecaster using precomputed attributions + similar markets.
        
        Args:
            train_data: List of markets with precomputed attributions
            valid_data: List of markets with precomputed attributions
            training_args: HuggingFace TrainingArguments
            
        Returns:
            Path to best model checkpoint
        """
        if self.model is None:
            self.setup_model()

        # Retrieve similar markets for each training/validation market
        train_similar = {
            m.market_id: self.retriever.find_similar(m) for m in tqdm(train_data, desc='Retrieving similar (train)')
        }
        valid_similar = {
            m.market_id: self.retriever.find_similar(m) for m in tqdm(valid_data, desc='Retrieving similar (valid)')
        }

        train_dataset = RAGMultiEventForecasterDataset(
            markets=train_data,
            similar_markets=train_similar,
            tokenizer=self.tokenizer,
            cache_dir=self.cache_dir,
        )
        valid_dataset = RAGMultiEventForecasterDataset(
            markets=valid_data,
            similar_markets=valid_similar,
            tokenizer=self.tokenizer,
            cache_dir=self.cache_dir,
        )

        trainer = WeightedTrainer(
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
        self,
        markets: List[MarketData],
        attributer: Any = None,
        batch_size: int = 8,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Predict market prices using RAG + attributions.
        
        Args:
            markets: List of markets with precomputed attributions (or use attributer)
            attributer: Optional PriorAttributer for on-the-fly attribution
            batch_size: Batch size for prediction
            
        Returns:
            List of prediction results
        """
        # Generate attributions if needed
        if attributer is not None:
            markets = self._generate_attributions(markets, attributer)
        
        # Retrieve similar markets
        similar_markets = {
            m.market_id: self.retriever.find_similar(m) for m in tqdm(markets, desc='Retrieving similar')
        }
        
        dataset = RAGMultiEventForecasterDataset(
            markets=markets,
            similar_markets=similar_markets,
            tokenizer=self.tokenizer,
            cache_dir=self.cache_dir,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._create_collate_fn(),
        )
        
        self.model.eval()
        results = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Predicting Batches'):
                input_ids = batch['input_ids'].to(self.model.llm.device)
                attention_mask = batch['attention_mask'].to(self.model.llm.device)
                labels = batch['labels'].to(self.model.llm.device)
                weights = batch['weights'].to(self.model.llm.device)
                group_ids = batch['group_ids'].to(self.model.llm.device)

                for group_idx in range(len(batch['event_ids'])):
                    group_mask = group_ids == group_idx
                    group_inputs = {
                        'input_ids': input_ids[group_mask],
                        'attention_mask': attention_mask[group_mask],
                    }

                    group_preds = self.model(**group_inputs).view(-1)
                    group_weights = weights[group_mask]
                    group_weights = group_weights / group_weights.sum()
                    weighted_pred = (group_preds * group_weights).sum()

                    results.append(
                        {
                            'event_id': batch['event_ids'][group_idx],
                            'market_id': batch['market_ids'][group_idx],
                            't': batch['ts'][group_idx],
                            'prediction': weighted_pred.item(),
                            'ground_truth': labels[group_mask][0].item(),
                        }
                    )
        return results
