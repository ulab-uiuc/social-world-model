"""
Train PriorAttributer using KL divergence from PosteriorAttributer.

The input data should have attributions precomputed using precompute_attributions.py
(which uses PosteriorAttributer). The PriorAttributer learns to predict the same
distribution without access to news.

Usage:
    python train_attributer.py \
        --train-data-path ../data/attributed/train.jsonl \
        --valid-data-path ../data/attributed/valid.jsonl \
        --output-dir ../saves/prior_attributer
"""
import argparse

from peft import LoraConfig
from transformers import TrainingArguments

from swm.attributer import BasicPriorAttributer
from swm.utils.utils import load_market_data, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train PriorAttributer using precomputed posterior attributions'
    )
    # Data paths
    parser.add_argument('--train-data-path', type=str, required=True,
                        help='Path to training data with attributions')
    parser.add_argument('--valid-data-path', type=str, required=True,
                        help='Path to validation data with attributions')
    
    # Model config
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen3-0.6B')
    parser.add_argument('--cache-dir', type=str, default='./cache')
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--max-seq-length', type=int, default=1024)
    
    # Training config
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--train-batch-size', type=int, default=8)
    parser.add_argument('--eval-batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-steps', type=int, default=0)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--logging-steps', type=int, default=100)
    parser.add_argument('--save-steps', type=int, default=500)
    parser.add_argument('--eval-steps', type=int, default=500)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Enable gradient checkpointing to save memory')
    
    # LoRA config
    parser.add_argument('--lora-alpha', type=float, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.1)
    parser.add_argument('--r', type=int, default=16)
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sanity-check', action='store_true')
    parser.add_argument('--max-news-per-bp', type=int, default=50,
                        help='Max news articles per breakpoint (default: 50)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Load data with precomputed attributions
    print(f"Loading training data from {args.train_data_path}...")
    train_data = load_market_data(args.train_data_path)
    print(f"Loading validation data from {args.valid_data_path}...")
    valid_data = load_market_data(args.valid_data_path)
    
    if args.sanity_check:
        train_data = train_data[:2]
        valid_data = valid_data[:2]
    
    # Helper to check if market has attributions in any breakpoint
    def has_attributions(market):
        if not market.daily_breakpoints:
            return False
        for bp in market.daily_breakpoints:
            if bp.get('attributions') and len(bp.get('attributions', [])) > 0:
                return True
        return False
    
    # Count breakpoints with attributions
    def count_breakpoints_with_attr(market):
        if not market.daily_breakpoints:
            return 0
        return sum(1 for bp in market.daily_breakpoints 
                   if bp.get('attributions') and len(bp.get('attributions', [])) > 0)
    
    # Check attributions are present
    train_with_attr = sum(1 for m in train_data if has_attributions(m))
    valid_with_attr = sum(1 for m in valid_data if has_attributions(m))
    train_bp_with_attr = sum(count_breakpoints_with_attr(m) for m in train_data)
    valid_bp_with_attr = sum(count_breakpoints_with_attr(m) for m in valid_data)
    
    print(f"Train: {train_with_attr}/{len(train_data)} markets have attributions ({train_bp_with_attr} breakpoints)")
    print(f"Valid: {valid_with_attr}/{len(valid_data)} markets have attributions ({valid_bp_with_attr} breakpoints)")
    
    if train_with_attr == 0:
        raise ValueError("No training data has attributions. Run step4_compute_posterior_attributions.py first.")
    
    # Initialize model
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
    )
    
    attributer = BasicPriorAttributer(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        max_seq_length=args.max_seq_length,
        lora_config=lora_config,
        gradient_checkpointing=args.gradient_checkpointing,
        max_news_per_bp=args.max_news_per_bp,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=f"prior_attributer_{args.model_name.split('/')[-1]}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy='steps',
        save_strategy='steps',
        fp16=args.fp16,
        metric_for_best_model='eval_loss',
        greater_is_better=False,  # Lower loss is better
        load_best_model_at_end=True,
        save_safetensors=False,
        remove_unused_columns=False,
        report_to='wandb',
        logging_first_step=True,
    )
    
    # Train using precomputed attributions (no need to pass posterior_attributer)
    best_checkpoint = attributer.train(
        train_data=train_data,
        valid_data=valid_data,
        training_args=training_args,
    )
    
    print(f"Best model saved to: {best_checkpoint}")
    attributer.save(best_checkpoint)


if __name__ == '__main__':
    main()

