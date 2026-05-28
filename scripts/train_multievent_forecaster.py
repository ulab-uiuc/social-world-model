"""
Train MultiEventForecaster using precomputed attributions.

The input data should have attributions precomputed using precompute_attributions.py

Single GPU:
    python train_multievent_forecaster.py \
        --train-data-path ../data/attributed/train.jsonl \
        --valid-data-path ../data/attributed/valid.jsonl \
        --output-dir ../saves/multievent_forecaster

Multi-GPU (DDP):
    torchrun --nproc_per_node=4 train_multievent_forecaster.py \
        --train-data-path ../data/attributed/train.jsonl \
        --valid-data-path ../data/attributed/valid.jsonl \
        --output-dir ../saves/multievent_forecaster
"""
import argparse
import os

import torch
import torch.distributed as dist
from transformers import TrainingArguments

from swm.forecaster import MultiEventForecaster
from swm.utils.utils import load_records, set_seed


def is_main_process():
    """Check if this is the main process in distributed training."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def setup_distributed():
    """Initialize distributed training if launched with torchrun."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        
        # Set device for this process
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    return 0, 1, 0


def print_main(*args, **kwargs):
    """Print only on main process."""
    if is_main_process():
        print(*args, **kwargs)


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
    parser.add_argument('--lr-scheduler-type', type=str, default='linear',
                        help='LR scheduler type (linear, cosine, cosine_with_restarts)')
    parser.add_argument('--logging-steps', type=int, default=100)
    parser.add_argument('--save-steps', type=int, default=100)
    parser.add_argument('--eval-steps', type=int, default=500)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Enable gradient checkpointing to save memory')
    
    parser.add_argument('--pooling-method', type=str, default='last_token',
                        choices=['last_token', 'mean'],
                        help='Pooling method for hidden states')


    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sanity-check', action='store_true')
    parser.add_argument('--overfit', action='store_true',
                        help='Overfit on a tiny subset to verify model can learn')
    parser.add_argument('--overfit-samples', type=int, default=4,
                        help='Number of samples to use for overfit test (default: 4)')
    parser.add_argument('--max-news', type=int, default=30,
                        help='Max news articles per record (default: 30)')
    parser.add_argument('--head-lr-multiplier', type=float, default=20.0,
                        help='LR multiplier for regression head (default: 20x base LR)')
    parser.add_argument('--null-subsample-ratio', type=float, default=1.0,
                        help='Fraction of null events (no positive attributions) to keep in TRAIN dataset. '
                             '1.0=keep all (default), <1.0 rebalances toward has-news. '
                             'Valid set always keeps ratio=1.0.')
    parser.add_argument('--window-std-threshold', type=float, default=0.0,
                        help='Drop records whose history price std < this (illiquid filter).'
                             ' 0.02 = drops ~28%% records on illiquid markets.')

    # Wandb
    parser.add_argument('--wandb-project', type=str, default='social-world-model',
                        help='Wandb project name')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    set_seed(args.seed)
    
    # Initialize wandb (only on main process)
    os.environ['WANDB_PROJECT'] = args.wandb_project
    if not is_main_process():
        os.environ['WANDB_DISABLED'] = 'true'
    
    print_main(f"Distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    
    print_main(f"Loading training data from {args.train_data_path}...")
    train_records = load_records(args.train_data_path)
    print_main(f"Loading validation data from {args.valid_data_path}...")
    valid_records = load_records(args.valid_data_path)

    if args.sanity_check:
        train_records = train_records[:2]
        valid_records = valid_records[:2]

    if args.overfit:
        print_main(f"[OVERFIT MODE] Using {args.overfit_samples} records for overfitting test")
        train_records = train_records[:args.overfit_samples]
        valid_records = train_records
        args.epochs = 100
        args.eval_steps = 10
        args.logging_steps = 1
        args.save_steps = 50
        args.learning_rate = 1e-4

    train_with_attr = sum(1 for r in train_records if r.attributions)
    valid_with_attr = sum(1 for r in valid_records if r.attributions)
    print_main(f"Train: {train_with_attr}/{len(train_records)} records have attributions")
    print_main(f"Valid: {valid_with_attr}/{len(valid_records)} records have attributions")

    if train_with_attr == 0:
        raise ValueError("No training records have attributions.")
    
    print_main("Full fine-tuning mode (no LoRA)")

    forecaster = MultiEventForecaster(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        gradient_checkpointing=args.gradient_checkpointing,
        max_news=args.max_news,
        head_lr_multiplier=args.head_lr_multiplier,
        pooling_method=args.pooling_method,
        null_subsample_ratio=args.null_subsample_ratio,
        window_std_threshold=args.window_std_threshold,
    )

    # HF constraint: load_best_model_at_end=True requires save_steps to be
    # a round multiple of eval_steps.
    if args.eval_steps > 0 and args.save_steps % args.eval_steps != 0:
        print_main(
            f"Adjusting save_steps from {args.save_steps} to {args.eval_steps} "
            "to satisfy load_best_model_at_end requirement."
        )
        args.save_steps = args.eval_steps
    
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
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy='steps',
        save_strategy='steps',
        fp16=args.fp16,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        load_best_model_at_end=True,
        save_safetensors=False,
        remove_unused_columns=False,
        report_to='wandb' if is_main_process() else 'none',
        run_name=f"{args.model_name.replace('/', '_')}_{args.output_dir.split('/')[-1]}",
        # DDP settings
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        local_rank=local_rank if world_size > 1 else -1,
    )
    
    best_checkpoint = forecaster.train(
        train_records=train_records,
        valid_records=valid_records,
        training_args=training_args,
    )
    
    print_main(f"Best model saved to: {best_checkpoint}")
    
    # Only save on main process
    if is_main_process():
        forecaster.save(best_checkpoint)
    
    # Cleanup distributed
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
