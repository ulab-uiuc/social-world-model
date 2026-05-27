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
    def __init__(self, *args, predict_absolute_price: bool = False, individual_loss: bool = False, loss_type: str = 'mse', head_lr_multiplier: float = 1.0, direction_loss_weight: float = 0.0, direction_only: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_absolute_price = predict_absolute_price
        self.individual_loss = individual_loss
        self.loss_type = loss_type
        self.head_lr_multiplier = head_lr_multiplier
        self.direction_loss_weight = direction_loss_weight
        self.direction_only = direction_only

    def create_optimizer(self):
        """Use a higher learning rate for the regression head."""
        if self.optimizer is not None:
            return self.optimizer

        model = self.model
        base_lr = self.args.learning_rate

        # Split params: regression_head vs everything else (LoRA + others)
        head_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'regression_head' in name:
                head_params.append(param)
            else:
                other_params.append(param)

        optimizer_grouped_parameters = [
            {'params': other_params, 'lr': base_lr},
            {'params': head_params, 'lr': base_lr * self.head_lr_multiplier},
        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        optimizer_kwargs.pop('lr', None)  # lr is set per-group
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def _process_group(
        self, model, inputs, weights, group_indices, group_label, is_prediction=False
    ):
        """Process a single group and return loss, prediction, and label.

        Args:
            group_indices: Indices in the flattened batch for this group
            group_label: The label for this group (already extracted)
        """
        group_size = len(group_indices)
        chunk_size = 8
        acc_pred = 0
        ind_loss = 0

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
                # Individual loss: each prediction independently targets the label
                if self.individual_loss and not is_prediction:
                    if self.loss_type == 'huber':
                        per_item_loss = torch.nn.functional.smooth_l1_loss(
                            chunk_preds, group_label.expand_as(chunk_preds), reduction='none'
                        )
                    else:
                        per_item_loss = (chunk_preds - group_label) ** 2
                    ind_loss += (per_item_loss * chunk_weights).sum()

        if self.individual_loss and not is_prediction:
            loss = ind_loss
        elif getattr(self, 'direction_only', False) and not is_prediction:
            # Direction-only mode: BCE on sign, no MSE regression. Only train on
            # |Δ| >= 0.02 events (skip flat-target groups, sign is noise there).
            if group_label.abs() >= 0.02:
                pred_prob = torch.sigmoid(acc_pred * 10)
                true_dir = (group_label > 0).float()
                loss = torch.nn.functional.binary_cross_entropy(pred_prob, true_dir)
            else:
                loss = torch.tensor(0.0, device=acc_pred.device, requires_grad=True)
        else:
            if self.loss_type == 'huber':
                loss = torch.nn.functional.smooth_l1_loss(acc_pred, group_label)
            else:
                loss = torch.nn.functional.mse_loss(acc_pred, group_label)

        # Auxiliary direction loss on top of MSE/Huber (only when not direction-only)
        if self.direction_loss_weight > 0 and not is_prediction and group_label.abs() > 1e-6 \
           and not getattr(self, 'direction_only', False):
            pred_prob = torch.sigmoid(acc_pred * 10)
            true_dir = (group_label > 0).float()
            dir_loss = torch.nn.functional.binary_cross_entropy(pred_prob, true_dir)
            loss = loss + self.direction_loss_weight * dir_loss

        return loss, acc_pred, group_label

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop('labels')
        weights = inputs.pop('weights')
        group_ids = inputs.pop('group_ids')

        total_loss = 0
        num_valid_groups = 0
        for group_idx, group in enumerate(torch.unique(group_ids)):
            group_indices = torch.where(group_ids == group)[0]
            # Use group_idx to get the correct label (labels are ordered by group)
            group_label = labels[group_idx]
            loss, _, _ = self._process_group(
                model, inputs, weights, group_indices, group_label
            )
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss
                num_valid_groups += 1

        return total_loss / max(num_valid_groups, 1)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        labels = inputs.pop('labels')  # Price delta (y_t+1 - y_t)
        weights = inputs.pop('weights')
        group_ids = inputs.pop('group_ids')
        before_prices = inputs.pop('before_prices')
        after_prices = inputs.pop('after_prices')

        all_losses, all_preds, all_labels = [], [], []
        for group_idx, group in enumerate(torch.unique(group_ids)):
            group_indices = torch.where(group_ids == group)[0]
            # Use group_idx to get the correct label
            group_label = labels[group_idx]
            loss, pred_delta, true_delta = self._process_group(
                model, inputs, weights, group_indices, group_label, is_prediction=True
            )

            if not torch.isnan(loss) and not torch.isinf(loss):
                all_losses.append(loss)
                # Return both delta and absolute price for comprehensive evaluation
                before_price = before_prices[group_idx]
                true_price = after_prices[group_idx]
                if self.predict_absolute_price:
                    # Model output IS the absolute price
                    pred_price = pred_delta  # misleading var name; it's abs price
                    real_pred_delta = pred_delta - before_price
                    real_true_delta = true_delta - before_price  # true_delta is also abs price here
                    all_preds.append(torch.stack([real_pred_delta, pred_price]))
                    all_labels.append(torch.stack([real_true_delta, true_price]))
                else:
                    pred_price = before_price + pred_delta
                    all_preds.append(torch.stack([pred_delta, pred_price]))
                    all_labels.append(torch.stack([true_delta, true_price]))

        return (
            torch.stack(all_losses).mean(),
            torch.stack(all_preds),  # Shape: (batch, 2) - [delta, price]
            torch.stack(all_labels),  # Shape: (batch, 2) - [delta, price]
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
        max_history_len: int = None,  # Max number of history points (None = use all)
        predict_absolute_price: bool = False,  # True for old checkpoints
        min_positive_attributions: int = 0,  # Min positive attributions to include sample
        individual_loss: bool = False,  # Train each news item independently
        loss_type: str = 'mse',  # 'mse' or 'huber'
        head_lr_multiplier: float = 1.0,  # LR multiplier for regression head
        min_max_attribution_score: float = 0.0,  # Only include breakpoints where max attr score >= this
        weight_temperature: float = 1.0,  # Temperature for softmax on attribution weights
        pooling_method: str = 'last_token',  # 'last_token' or 'mean'
        direction_loss_weight: float = 0.0,  # Weight for direction classification loss (0 = off)
        null_subsample_ratio: float = 1.0,  # Fraction of null events to keep in TRAIN dataset (rebalance)
        direction_only: bool = False,  # Pure direction classifier (BCE on sign, no MSE)
        only_null: bool = False,  # Train on null events ONLY (time-series LM baseline)
        window_std_threshold: float = 0.0,  # Drop events whose pre-BP price window std < this
        null_max_delta: float = 0.0,  # If >0, drop null events with |Δp| > this
        null_exclude_breakpoint: bool = False,  # Drop null events where sample_type=='breakpoint'
        flip_augment: bool = False,  # Emit 1-p mirror of each event (debias)
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_seq_length = max_seq_length
        self.model = None
        self.cache_dir = Path(cache_dir)
        self.lora_config = lora_config
        self.gradient_checkpointing = gradient_checkpointing
        self.max_news_per_bp = max_news_per_bp
        self.pooling_method = pooling_method
        self.max_history_len = max_history_len
        self.predict_absolute_price = predict_absolute_price
        self.min_positive_attributions = min_positive_attributions
        self.individual_loss = individual_loss
        self.loss_type = loss_type
        self.head_lr_multiplier = head_lr_multiplier
        self.min_max_attribution_score = min_max_attribution_score
        self.weight_temperature = weight_temperature
        self.direction_loss_weight = direction_loss_weight
        self.null_subsample_ratio = null_subsample_ratio
        self.direction_only = direction_only
        self.only_null = only_null
        self.window_std_threshold = window_std_threshold
        self.null_max_delta = null_max_delta
        self.null_exclude_breakpoint = null_exclude_breakpoint
        self.flip_augment = flip_augment

    def setup_model(self) -> None:
        config = LLMRegressorConfig(
            base_model_name_or_path=self.model_name,
            max_length=self.max_seq_length,
            pooling_method=self.pooling_method,
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
            if self.predict_absolute_price:
                # Old format: pad to at least max_seq_length
                max_len = max(
                    max(x['input_ids'].size(-1) for x in batch), self.max_seq_length
                )
            else:
                # New format: cap at max_seq_length
                max_len = min(
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
            all_before_prices = []
            all_after_prices = []

            for group_idx, item in enumerate(batch):
                if self.predict_absolute_price:
                    # Old format: pad without truncation
                    padded_inputs = torch.nn.functional.pad(
                        item['input_ids'],
                        (0, max_len - item['input_ids'].size(-1)),
                        value=self.tokenizer.pad_token_id,
                    )
                else:
                    padded_inputs = torch.nn.functional.pad(
                        item['input_ids'][:, :max_len],
                        (0, max_len - min(item['input_ids'].size(-1), max_len)),
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
                all_before_prices.append(item['before_price'])
                all_after_prices.append(item['after_price'])

            return {
                'input_ids': torch.cat(all_input_ids),
                'attention_mask': torch.cat(all_attention_masks),
                'labels': torch.stack(all_labels),  # Price delta (y_t+1 - y_t)
                'weights': torch.cat(all_weights),
                'group_ids': torch.cat(all_group_ids),
                'market_ids': all_market_ids,
                'event_ids': all_event_ids,
                'ts': all_ts,
                'before_prices': torch.stack(all_before_prices),  # For recovering absolute price
                'after_prices': torch.stack(all_after_prices),  # Ground truth absolute price
            }

        return collate_fn

    def _safe_train(self, trainer):
        """Run training, catching FileNotFoundError from load_best_model_at_end with LoRA."""
        try:
            trainer.train()
        except FileNotFoundError:
            # LoRA adapter saves as adapter_model.safetensors, not pytorch_model.bin
            # load_best_model_at_end may fail; the best checkpoint is still saved on disk
            pass

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
            max_seq_length=self.max_seq_length,
            max_history_len=self.max_history_len,
            predict_absolute_price=self.predict_absolute_price,
            min_positive_attributions=self.min_positive_attributions,
            min_max_attribution_score=self.min_max_attribution_score,
            weight_temperature=self.weight_temperature,
            null_subsample_ratio=self.null_subsample_ratio,
            only_null=self.only_null,
            window_std_threshold=self.window_std_threshold,
            null_max_delta=self.null_max_delta,
            null_exclude_breakpoint=self.null_exclude_breakpoint,
            flip_augment=self.flip_augment,
        )
        valid_dataset = MultiEventForecasterDataset(
            markets=valid_data,
            tokenizer=self.tokenizer,
            cache_dir=self.cache_dir,
            max_news_per_bp=self.max_news_per_bp,
            max_seq_length=self.max_seq_length,
            max_history_len=self.max_history_len,
            predict_absolute_price=self.predict_absolute_price,
            min_positive_attributions=self.min_positive_attributions,
            min_max_attribution_score=self.min_max_attribution_score,
            weight_temperature=self.weight_temperature,
            null_subsample_ratio=1.0,  # never subsample valid
            only_null=self.only_null,
            window_std_threshold=self.window_std_threshold,
            null_max_delta=self.null_max_delta,
            null_exclude_breakpoint=self.null_exclude_breakpoint,
            flip_augment=self.flip_augment,
        )

        def compute_metrics(eval_pred):
            """Compute both delta MSE and absolute price MSE."""
            predictions, labels = eval_pred
            # predictions/labels shape: (batch, 2) - [delta, price]
            pred_delta = predictions[:, 0]
            pred_price = predictions[:, 1]
            true_delta = labels[:, 0]
            true_price = labels[:, 1]
            return {
                'delta_mse': mean_squared_error(true_delta, pred_delta),
                'price_mse': mean_squared_error(true_price, pred_price),
            }

        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=self._create_collate_fn(),
            compute_metrics=compute_metrics,
            predict_absolute_price=self.predict_absolute_price,
            individual_loss=self.individual_loss,
            loss_type=self.loss_type,
            head_lr_multiplier=self.head_lr_multiplier,
            direction_loss_weight=self.direction_loss_weight,
            direction_only=self.direction_only,
        )

        self._safe_train(trainer)
        if trainer.state.best_model_checkpoint:
            return trainer.state.best_model_checkpoint
        final_model_dir = Path(training_args.output_dir) / 'final-model'
        trainer.save_model(final_model_dir)
        return str(final_model_dir)

    def predict(
        self,
        markets: List[MarketData],
        attributer: Any = None,
        batch_size: int = 8,
        score_threshold: float = 0.0,
        top_k: int = 0,
        weight_temperature: float = 1.0,
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Predict market prices.

        Args:
            markets: List of markets. Should have either:
                - Precomputed attributions in market.attributions, OR
                - Pass an attributer to generate attributions on-the-fly
            attributer: Optional PriorAttributer to generate attributions for inference
            batch_size: Batch size for prediction
            score_threshold: Drop attributed news below this score
            top_k: Keep only top-k attributed news (0 = keep all)
            weight_temperature: Temperature for softmax on attribution weights (lower = sharper)

        Returns:
            List of prediction results
        """
        # If attributer provided, generate attributions on-the-fly
        if attributer is not None:
            markets = self._generate_attributions(
                markets, attributer,
                score_threshold=score_threshold,
                top_k=top_k,
            )

        dataset = MultiEventForecasterDataset(
            markets=markets,
            tokenizer=self.tokenizer,
            cache_dir=self.cache_dir,
            max_news_per_bp=self.max_news_per_bp,
            max_seq_length=self.max_seq_length,
            max_history_len=self.max_history_len,
            predict_absolute_price=self.predict_absolute_price,
            min_positive_attributions=self.min_positive_attributions,
            weight_temperature=weight_temperature,
            only_null=self.only_null,
            window_std_threshold=self.window_std_threshold,
            null_max_delta=self.null_max_delta,
            null_exclude_breakpoint=self.null_exclude_breakpoint,
            flip_augment=self.flip_augment,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._create_collate_fn(),
        )
        self.model.eval()
        results = []
        chunk_size = 8  # Same as training for consistency
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Predicting Batches'):
                input_ids = batch['input_ids'].to(self.model.llm.device)
                attention_mask = batch['attention_mask'].to(self.model.llm.device)
                labels = batch['labels'].to(self.model.llm.device)
                weights = batch['weights'].to(self.model.llm.device)
                group_ids = batch['group_ids'].to(self.model.llm.device)
                before_prices = batch['before_prices'].to(self.model.llm.device)
                after_prices = batch['after_prices'].to(self.model.llm.device)

                for group_idx in range(len(batch['event_ids'])):
                    group_mask = group_ids == group_idx
                    group_indices = torch.where(group_mask)[0]
                    group_size = len(group_indices)

                    # Normalize weights with epsilon (same as training)
                    group_weights = weights[group_mask]
                    normalized_weights = group_weights / (group_weights.sum() + 1e-8)

                    # Process in chunks (same as training) for memory efficiency
                    acc_pred = 0.0
                    for i in range(0, group_size, chunk_size):
                        chunk_indices = group_indices[i : i + chunk_size]
                        chunk_inputs = {
                            'input_ids': input_ids[chunk_indices],
                            'attention_mask': attention_mask[chunk_indices],
                        }
                        chunk_weights = normalized_weights[i : i + chunk_size]

                        chunk_preds = self.model(**chunk_inputs).view(-1)
                        acc_pred += (chunk_preds * chunk_weights).sum()

                    before_price = before_prices[group_idx].item()
                    true_price = after_prices[group_idx].item()

                    if self.predict_absolute_price:
                        # Old checkpoints: model output IS the absolute price
                        pred_price = acc_pred.item()
                        pred_delta = pred_price - before_price
                        true_delta = true_price - before_price
                    else:
                        # New checkpoints: model output is the delta
                        pred_delta = acc_pred.item()
                        pred_price = before_price + pred_delta
                        true_delta = labels[group_idx].item()

                    results.append(
                        {
                            'event_id': batch['event_ids'][group_idx],
                            'market_id': batch['market_ids'][group_idx],
                            't': batch['ts'][group_idx],
                            'pred_delta': pred_delta,
                            'true_delta': true_delta,
                            'pred_price': pred_price,
                            'true_price': true_price,
                            'before_price': before_price,
                        }
                    )
        return results

    def _generate_attributions(
        self,
        markets: List[MarketData],
        attributer: Any,
        score_threshold: float = 0.0,
        top_k: int = 0,
    ) -> List[MarketData]:
        """
        Generate attributions for markets using the provided attributer.

        Args:
            markets: List of markets with daily_breakpoints containing news
            attributer: Trained BasicPriorAttributer for real-time attribution
            score_threshold: Drop news with attribution score below this
            top_k: Keep only top-k news by score (0 = keep all above threshold)

        Returns:
            Markets with attributions added to each breakpoint
        """
        total_bps = 0
        total_news_kept = 0
        total_news_total = 0
        for market in tqdm(markets, desc='Generating attributions'):
            if not market.daily_breakpoints:
                continue

            for bp in market.daily_breakpoints:
                # CRITICAL: when an attributer is provided we must use ITS
                # output exclusively. The test file may carry preexisting
                # (oracle) attributions; clear them up-front so that ANY path
                # which produces no attributions (null-gate returning [],
                # all news filtered, <2 news) yields a truly empty/no-news
                # breakpoint instead of silently leaking the oracle labels.
                bp['attributions'] = []

                news_list = bp.get('news', [])
                if not news_list or len(news_list) < 2:
                    continue

                attributions = attributer.attribute_breakpoint(
                    market, bp,
                    score_threshold=score_threshold,
                    top_k=top_k,
                )

                total_news_total += len(news_list)
                if attributions:
                    bp['attributions'] = [
                        {'news_idx': attr['news_idx'], 'score': attr['score']}
                        for attr in attributions
                    ]
                    total_news_kept += len(attributions)
                    total_bps += 1

        if total_news_total > 0:
            print(f"Generated attributions for {total_bps} breakpoints, "
                  f"kept {total_news_kept}/{total_news_total} news "
                  f"({total_news_kept/total_news_total*100:.1f}%)")
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
            max_seq_length=self.max_seq_length,
        )
        valid_dataset = RAGMultiEventForecasterDataset(
            markets=valid_data,
            similar_markets=valid_similar,
            tokenizer=self.tokenizer,
            cache_dir=self.cache_dir,
            max_seq_length=self.max_seq_length,
        )

        def compute_metrics(eval_pred):
            """Compute both delta MSE and absolute price MSE."""
            predictions, labels = eval_pred
            # predictions/labels shape: (batch, 2) - [delta, price]
            pred_delta = predictions[:, 0]
            pred_price = predictions[:, 1]
            true_delta = labels[:, 0]
            true_price = labels[:, 1]
            return {
                'delta_mse': mean_squared_error(true_delta, pred_delta),
                'price_mse': mean_squared_error(true_price, pred_price),
            }

        trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=self._create_collate_fn(),
            compute_metrics=compute_metrics,
            predict_absolute_price=self.predict_absolute_price,
            individual_loss=self.individual_loss,
            loss_type=self.loss_type,
            head_lr_multiplier=self.head_lr_multiplier,
            direction_loss_weight=self.direction_loss_weight,
            direction_only=self.direction_only,
        )

        self._safe_train(trainer)
        if trainer.state.best_model_checkpoint:
            return trainer.state.best_model_checkpoint
        final_model_dir = Path(training_args.output_dir) / 'final-model'
        trainer.save_model(final_model_dir)
        return str(final_model_dir)

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
            max_seq_length=self.max_seq_length,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._create_collate_fn(),
        )
        
        self.model.eval()
        results = []
        chunk_size = 8  # Same as training for consistency
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Predicting Batches'):
                input_ids = batch['input_ids'].to(self.model.llm.device)
                attention_mask = batch['attention_mask'].to(self.model.llm.device)
                labels = batch['labels'].to(self.model.llm.device)
                weights = batch['weights'].to(self.model.llm.device)
                group_ids = batch['group_ids'].to(self.model.llm.device)
                before_prices = batch['before_prices'].to(self.model.llm.device)
                after_prices = batch['after_prices'].to(self.model.llm.device)

                for group_idx in range(len(batch['event_ids'])):
                    group_mask = group_ids == group_idx
                    group_indices = torch.where(group_mask)[0]
                    group_size = len(group_indices)
                    
                    # Normalize weights with epsilon (same as training)
                    group_weights = weights[group_mask]
                    normalized_weights = group_weights / (group_weights.sum() + 1e-8)
                    
                    # Process in chunks (same as training) for memory efficiency
                    acc_pred = 0.0
                    for i in range(0, group_size, chunk_size):
                        chunk_indices = group_indices[i : i + chunk_size]
                        chunk_inputs = {
                            'input_ids': input_ids[chunk_indices],
                            'attention_mask': attention_mask[chunk_indices],
                        }
                        chunk_weights = normalized_weights[i : i + chunk_size]
                        
                        chunk_preds = self.model(**chunk_inputs).view(-1)
                        acc_pred += (chunk_preds * chunk_weights).sum()
                    
                    pred_delta = acc_pred
                    
                    # Recover absolute price from predicted delta
                    before_price = before_prices[group_idx].item()
                    pred_price = before_price + pred_delta.item()
                    true_price = after_prices[group_idx].item()
                    true_delta = labels[group_idx].item()

                    results.append(
                        {
                            'event_id': batch['event_ids'][group_idx],
                            'market_id': batch['market_ids'][group_idx],
                            't': batch['ts'][group_idx],
                            'pred_delta': pred_delta.item(),
                            'true_delta': true_delta,
                            'pred_price': pred_price,
                            'true_price': true_price,
                            'before_price': before_price,
                        }
                    )
        return results
