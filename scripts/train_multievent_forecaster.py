"""
Train MultiEventForecaster using precomputed attributions.

The input data should have attributions precomputed using precompute_attributions.py

Usage:
    python train_multievent_forecaster.py \
        --train-data-path ../data/attributed/train.jsonl \
        --valid-data-path ../data/attributed/valid.jsonl \
        --output-dir ../saves/multievent_forecaster
"""
import argparse

from peft import LoraConfig
from transformers import TrainingArguments

from swm.forecaster import MultiEventForecaster
from swm.utils.utils import load_market_data, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train MultiEventForecaster with precomputed attributions'
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
    parser.add_argument('--save-steps', type=int, default=100)
    parser.add_argument('--eval-steps', type=int, default=500)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Enable gradient checkpointing to save memory')
    
    # LoRA config
    parser.add_argument('--lora-alpha', type=float, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.1)
    parser.add_argument('--r', type=int, default=16)
    parser.add_argument('--target-modules', type=str, default='q_proj,v_proj,k_proj,o_proj',
                        help='Comma-separated LoRA target modules')
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sanity-check', action='store_true')
    parser.add_argument('--max-news-per-bp', type=int, default=30,
                        help='Max news articles per breakpoint (default: 30)')
    
    # Wandb
    parser.add_argument('--wandb-project', type=str, default='social-world-model',
                        help='Wandb project name')
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Initialize wandb
    import os
    os.environ['WANDB_PROJECT'] = args.wandb_project
    
    # Load data with precomputed attributions
    print(f"Loading training data from {args.train_data_path}...")
    train_data = load_market_data(args.train_data_path)
    print(f"Loading validation data from {args.valid_data_path}...")
    valid_data = load_market_data(args.valid_data_path)
    
    if args.sanity_check:
        train_data = train_data[:2]
        valid_data = valid_data[:2]
    
    # Check attributions are present (attributions are stored in breakpoints, not on market directly)
    def count_bp_with_attr(markets):
        total_markets = 0
        total_bps = 0
        for m in markets:
            if not m.daily_breakpoints:
                continue
            has_attr = False
            for bp in m.daily_breakpoints:
                if bp.get('attributions'):
                    total_bps += 1
                    has_attr = True
            if has_attr:
                total_markets += 1
        return total_markets, total_bps
    
    train_markets, train_bps = count_bp_with_attr(train_data)
    valid_markets, valid_bps = count_bp_with_attr(valid_data)
    print(f"Train: {train_markets}/{len(train_data)} markets have attributions ({train_bps} breakpoints)")
    print(f"Valid: {valid_markets}/{len(valid_data)} markets have attributions ({valid_bps} breakpoints)")
    
    if train_bps == 0:
        raise ValueError("No training data has attributions. Run step4_compute_posterior_attributions.py first.")
    
    # Initialize model
    target_modules = [m.strip() for m in args.target_modules.split(',')]
    print(f"LoRA config: r={args.r}, target_modules={target_modules}")
    
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
    )
    
    forecaster = MultiEventForecaster(
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
        metric_for_best_model='loss',
        save_safetensors=False,
        remove_unused_columns=False,
        report_to='wandb',
        run_name=f"{args.model_name.replace('/', '_')}_{args.output_dir.split('/')[-1]}",
    )
    
    # Train
    best_checkpoint = forecaster.train(
        train_data=train_data,
        valid_data=valid_data,
        training_args=training_args,
    )
    
    print(f"Best model saved to: {best_checkpoint}")
    forecaster.save(best_checkpoint)


if __name__ == '__main__':
    main()

