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
from typing import Any, Dict

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
    parser.add_argument('--max-steps', type=int, default=-1,
                        help='Cap total optimizer steps (-1=disabled). For quick save-path smoke tests.')
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
    parser.add_argument('--bf16', action='store_true',
                        help='Mixed-precision bf16 training. A100/H100 only.')
    parser.add_argument('--fsdp', type=str, default='',
                        help='FSDP mode, e.g. "full_shard auto_wrap". Empty disables FSDP.')
    parser.add_argument('--fsdp-min-num-params', type=float, default=0,
                        help='If >0, use size_based FSDP auto-wrap (wraps ANY module > this many params, incl. the embedding -> gathered on save). Overrides transformer-layer-cls.')
    parser.add_argument('--fsdp-transformer-layer-cls', type=str, default='',
                        help='Transformer layer class for FSDP auto-wrap, e.g. Qwen3DecoderLayer.')
    parser.add_argument('--lora-r', type=int, default=0,
                        help='LoRA rank. 0 disables LoRA (full FT).')
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--lora-target-modules', type=str,
                        default='q_proj,k_proj,v_proj,o_proj',
                        help='Comma-separated module names for LoRA target.')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Enable gradient checkpointing to save memory')
    
    parser.add_argument('--pooling-method', type=str, default='last_token',
                        choices=['last_token', 'mean'],
                        help='Pooling method for hidden states')
    parser.add_argument('--predict-delta', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Predict target.p - before_price (delta) instead of '
                             'absolute price. Default: True. Use --no-predict-delta '
                             'for the old absolute-price target.')


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
    parser.add_argument('--max-history-len', type=int, default=None,
                        help='Trim each record history to the last N days BEFORE training. '
                             'Small N (e.g. 1) removes the momentum trajectory, forcing the '
                             'model to read news for the delta. before_price is unchanged.')
    parser.add_argument('--delta-weighted-loss', action='store_true',
                        help='Weight each record loss by |true_delta|(+floor): big moves '
                             'dominate the gradient, fighting MSE mean-collapse.')
    parser.add_argument('--delta-weight-floor', type=float, default=0.0,
                        help='Floor added to |true_delta| weight (so small moves are not '
                             'fully ignored). Default 0 (pure |delta|).')
    parser.add_argument('--mae-loss', action='store_true',
                        help='L1/MAE loss (optimizes conditional median) instead of MSE. '
                             'De-shrinks for MAE directly; overrides vol/delta-weighted.')
    parser.add_argument('--per-news-loss', action='store_true',
                        help='Responsibility-weighted L_wm: score each news vs the move then weight by attribution (vs error-of-weighted-mean). Composes with --vol-normalized-loss.')
    parser.add_argument('--huber-loss', action='store_true',
                        help='Huber/smooth-L1 loss (robust; quadratic near 0, linear in tail).')
    parser.add_argument('--huber-beta', type=float, default=0.05,
                        help='Huber transition point (default 0.05 ~ median move scale).')
    parser.add_argument('--attr-noise-drop', type=float, default=0.0,
                        help='H3: drop each gold-attributed news w.p. this (false negatives).')
    parser.add_argument('--attr-noise-add', type=float, default=0.0,
                        help='H3: add distractor negatives = round(this*n_pos) at small weight (false positives).')
    parser.add_argument('--prior-attr-path', type=str, default=None,
                        help='Prior-attribution dump (inference_prior_attribution.py output) on the TRAIN set; '
                             'mixed into posterior weights to bridge train/inference distribution.')
    parser.add_argument('--post-prior-mix', type=float, default=1.0,
                        help='Blend a in w=a*posterior+(1-a)*prior. 1.0=pure posterior (default), 0.0=pure prior, 0.5=balanced.')
    parser.add_argument('--read-all', action='store_true',
                        help='Read ALL candidate news jointly in one prompt (vs per-news weighted avg).')
    parser.add_argument('--bounded-output', action='store_true',
                        help='Sigmoid the head -> per-news pred in [0,1]=target_p; delta=target_p-before '
                             'respects [-bp,1-bp] bounds. Implies --no-predict-delta (label=target_p).')
    parser.add_argument('--vol-normalized-loss', action='store_true',
                        help='Loss = ((pred-delta)/vol)^2 = predict the move as a multiple '
                             'of historical volatility. Standardizes scale across markets, '
                             'fights mean-collapse. Overrides --delta-weighted-loss.')
    parser.add_argument('--vol-floor', type=float, default=0.01,
                        help='Floor on historical volatility in the vol-normalized loss '
                             '(avoids blow-up when a market was flat). Default 0.01.')
    parser.add_argument('--train-attributed-only', action='store_true',
                        help='Train/eval only on records that have >=1 positive-score '
                             'attribution (news-driven events), so the gradient is not '
                             'dominated by null/predict-mean records.')
    parser.add_argument('--null-subsample-ratio', type=float, default=1.0,
                        help='Fraction of null events (no positive attributions) to keep in TRAIN dataset. '
                             '1.0=keep all (default), <1.0 rebalances toward has-news. '
                             'Valid set always keeps ratio=1.0.')

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

    def _has_pos_attr(r):
        n = len(r.news or [])
        return any(0 <= a.get('news_idx', -1) < n and float(a.get('score') or 0) > 0
                   for a in (r.attributions or []))

    if args.train_attributed_only:
        b_tr, b_va = len(train_records), len(valid_records)
        train_records = [r for r in train_records if _has_pos_attr(r)]
        valid_records = [r for r in valid_records if _has_pos_attr(r)]
        print_main(f"[attributed-only] train {b_tr}->{len(train_records)}, valid {b_va}->{len(valid_records)}")

    if args.max_history_len is not None:
        for r in train_records + valid_records:
            if r.history and len(r.history) > args.max_history_len:
                r.history = r.history[-args.max_history_len:]
        print_main(f"[max-history-len] trimmed history to last {args.max_history_len} day(s)")

    train_with_attr = sum(1 for r in train_records if r.attributions)
    valid_with_attr = sum(1 for r in valid_records if r.attributions)
    print_main(f"Train: {train_with_attr}/{len(train_records)} records have attributions")
    print_main(f"Valid: {valid_with_attr}/{len(valid_records)} records have attributions")

    if train_with_attr == 0:
        raise ValueError("No training records have attributions.")
    
    lora_config = None
    if args.lora_r > 0:
        from peft import LoraConfig, TaskType
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(','),
            bias='none',
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        print_main(f"LoRA mode: r={args.lora_r}, alpha={args.lora_alpha}, "
                   f"targets={args.lora_target_modules}")
    else:
        print_main("Full fine-tuning mode (no LoRA)")

    forecaster = MultiEventForecaster(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        gradient_checkpointing=args.gradient_checkpointing,
        max_news=args.max_news,
        head_lr_multiplier=args.head_lr_multiplier,
        pooling_method=args.pooling_method,
        null_subsample_ratio=args.null_subsample_ratio,
        predict_delta=args.predict_delta,
        delta_weighted=args.delta_weighted_loss,
        vol_normalized=args.vol_normalized_loss,
        vol_floor=args.vol_floor,
        mae_loss=args.mae_loss,
        huber_loss=args.huber_loss,
        per_news_loss=args.per_news_loss,
        huber_beta=args.huber_beta,
        bounded_output=args.bounded_output,
        read_all=args.read_all,
        attr_noise_drop=args.attr_noise_drop,
        attr_noise_add=args.attr_noise_add,
        prior_attr_path=args.prior_attr_path,
        post_prior_mix=args.post_prior_mix,
        delta_weight_floor=args.delta_weight_floor,
        lora_config=lora_config,
    )

    # HF constraint: load_best_model_at_end=True requires save_steps to be
    # a round multiple of eval_steps.
    if args.eval_steps > 0 and args.save_steps % args.eval_steps != 0:
        print_main(
            f"Adjusting save_steps from {args.save_steps} to {args.eval_steps} "
            "to satisfy load_best_model_at_end requirement."
        )
        args.save_steps = args.eval_steps
    
    fsdp_kwargs: Dict[str, Any] = {}
    if args.fsdp:
        fsdp_kwargs['fsdp'] = args.fsdp
        # FULL_STATE_DICT: gather the unsharded weights on save. Without it, the
        # root FSDP unit (embeddings + final norm) is written as a flat shard
        # (e.g. embed_tokens.weight saved as 1-D [hidden*vocab/world_size]),
        # which can't be reloaded — only the auto-wrapped decoder layers survive.
        fsdp_config: Dict[str, Any] = {'state_dict_type': 'FULL_STATE_DICT'}
        if args.fsdp_min_num_params and args.fsdp_min_num_params > 0:
            fsdp_config['min_num_params'] = int(args.fsdp_min_num_params)  # size-based: wraps embedding too
        elif args.fsdp_transformer_layer_cls:
            fsdp_config['transformer_layer_cls_to_wrap'] = [args.fsdp_transformer_layer_cls]
        fsdp_kwargs['fsdp_config'] = fsdp_config

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
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
        bf16=args.bf16,
        **fsdp_kwargs,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        # load_best_model_at_end=True would torch.load at training end, which
        # tripped HF's CVE-2025-32434 check on torch<2.6. Skip the auto-reload;
        # `trainer.state.best_model_checkpoint` is still populated.
        load_best_model_at_end=False,
        save_safetensors=True,
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
