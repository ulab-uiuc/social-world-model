"""
Train MultiEventWorldModel using precomputed attributions.

The input data should have attributions precomputed using precompute_attributions.py

Single GPU:
    python train_multievent_world_model.py \
        --train-data-path ../data/attributed/train.jsonl \
        --valid-data-path ../data/attributed/valid.jsonl \
        --output-dir ../saves/multievent_world_model

Multi-GPU (DDP):
    torchrun --nproc_per_node=4 train_multievent_world_model.py \
        --train-data-path ../data/attributed/train.jsonl \
        --valid-data-path ../data/attributed/valid.jsonl \
        --output-dir ../saves/multievent_world_model
"""

import argparse
import os
from typing import Any, Dict

import torch
import torch.distributed as dist
from transformers import TrainingArguments

from swm.utils.utils import load_records, set_seed
from swm.world_model import MultiEventWorldModel


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

        # Initialize process group. Use a long timeout: the end-of-training
        # best-model / final-model save does an FSDP FULL_STATE_DICT gather with
        # offload_to_cpu (slow for 3B), during which idle ranks would otherwise
        # hit the default 10-min NCCL watchdog and SIGABRT the whole job.
        if not dist.is_initialized():
            from datetime import timedelta
            dist.init_process_group(
                backend='nccl', timeout=timedelta(minutes=60)
            )

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
        description='Train MultiEventWorldModel with precomputed attributions'
    )
    # Data paths
    parser.add_argument(
        '--train-data-path',
        type=str,
        required=True,
        help='Path to training data with attributions',
    )
    parser.add_argument(
        '--valid-data-path',
        type=str,
        required=True,
        help='Path to validation data with attributions',
    )

    # Model config
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen3-0.6B')
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--max-seq-length', type=int, default=1024)

    # Training config
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument(
        '--max-steps',
        type=int,
        default=-1,
        help='Cap total optimizer steps (-1=disabled). For quick save-path smoke tests.',
    )
    parser.add_argument('--train-batch-size', type=int, default=8)
    parser.add_argument('--eval-batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-steps', type=int, default=0)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument(
        '--lr-scheduler-type',
        type=str,
        default='linear',
        help='LR scheduler type (linear, cosine, cosine_with_restarts)',
    )
    parser.add_argument('--logging-steps', type=int, default=100)
    parser.add_argument('--save-steps', type=int, default=100)
    parser.add_argument(
        '--save-total-limit',
        type=int,
        default=None,
        help='Cap number of saved checkpoints (HF keeps the most '
        'recent N + the best). Use for full-FT (each ckpt is '
        'GBs) to avoid filling the disk. None=keep all.',
    )
    parser.add_argument(
        '--save-only-model',
        action='store_true',
        help='Save ONLY model weights (skip optimizer/scheduler/rng '
        'state). For full-FT this cuts checkpoint size ~3-4x '
        '(no fp32 Adam state) and the slow FSDP optimizer gather. '
        'Cannot resume training from these, but we only need the '
        'final weights for inference.',
    )
    parser.add_argument('--eval-steps', type=int, default=500)
    parser.add_argument(
        '--no-mid-checkpoints',
        action='store_true',
        help='Disable mid-training checkpointing (save_strategy=no). '
        'Forces the run to save ONCE at the end via '
        'trainer.save_model(), which under FSDP goes through '
        'accelerate.get_state_dict (correct gather of the root '
        'embedding). Mid-training _save_checkpoint uses '
        'save_fsdp_model which writes a FLAT embedding shard that '
        'cannot be reloaded. Use for full-FT FSDP runs.',
    )
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument(
        '--bf16',
        action='store_true',
        help='Mixed-precision bf16 training. A100/H100 only.',
    )
    parser.add_argument(
        '--fsdp',
        type=str,
        default='',
        help='FSDP mode, e.g. "full_shard auto_wrap". Empty disables FSDP.',
    )
    parser.add_argument(
        '--fsdp-min-num-params',
        type=float,
        default=0,
        help='If >0, use size_based FSDP auto-wrap (wraps ANY module > this many params, incl. the embedding -> gathered on save). Overrides transformer-layer-cls.',
    )
    parser.add_argument(
        '--fsdp-transformer-layer-cls',
        type=str,
        default='',
        help='Transformer layer class for FSDP auto-wrap, e.g. Qwen3DecoderLayer.',
    )
    parser.add_argument(
        '--gradient-checkpointing',
        action='store_true',
        help='Enable gradient checkpointing to save memory',
    )

    parser.add_argument(
        '--pooling-method',
        type=str,
        default='last_token',
        choices=['last_token', 'mean'],
        help='Pooling method for hidden states',
    )
    parser.add_argument(
        '--predict-delta',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Predict target.p - before_price (delta) instead of '
        'absolute price. Default: True. Use --no-predict-delta '
        'for the old absolute-price target.',
    )

    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sanity-check', action='store_true')
    parser.add_argument(
        '--overfit',
        action='store_true',
        help='Overfit on a tiny subset to verify model can learn',
    )
    parser.add_argument(
        '--overfit-samples',
        type=int,
        default=4,
        help='Number of samples to use for overfit test (default: 4)',
    )
    parser.add_argument(
        '--max-news',
        type=int,
        default=30,
        help='Max news articles per record (default: 30)',
    )
    parser.add_argument(
        '--head-lr-multiplier',
        type=float,
        default=20.0,
        help='LR multiplier for regression head (default: 20x base LR)',
    )
    parser.add_argument(
        '--max-history-len',
        type=int,
        default=None,
        help='Trim each record history to the last N days BEFORE training. '
        'Small N (e.g. 1) removes the momentum trajectory, forcing the '
        'model to read news for the delta. before_price is unchanged.',
    )
    parser.add_argument(
        '--null-rho0',
        type=float,
        default=1.0,
        help='Null event raw mass o_0=rho0 in the odds categorical.',
    )
    parser.add_argument(
        '--odds-eps',
        type=float,
        default=1e-3,
        help='Epsilon in odds o_i=(a+eps)/(1-a+eps).',
    )
    parser.add_argument(
        '--odds-temp',
        type=float,
        default=1.0,
        help='Smoothing temperature: o_i^(1/T). T>1 flattens. odds already '
        'spreads the distribution so T=1 is the default (no extra sharpen).',
    )
    parser.add_argument(
        '--train-attributed-only',
        action='store_true',
        help='Train/eval only on records that have >=1 positive-score '
        'attribution (news-driven events), so the gradient is not '
        'dominated by null/predict-mean records.',
    )
    parser.add_argument(
        '--null-subsample-ratio',
        type=float,
        default=1.0,
        help='Fraction of null events (no positive attributions) to keep in TRAIN dataset. '
        '1.0=keep all (default), <1.0 rebalances toward has-news. '
        'Valid set always keeps ratio=1.0.',
    )

    # Wandb
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='social-world-model',
        help='Wandb project name',
    )

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

    print_main(
        f'Distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}'
    )

    print_main(f'Loading training data from {args.train_data_path}...')
    train_records = load_records(args.train_data_path)
    print_main(f'Loading validation data from {args.valid_data_path}...')
    valid_records = load_records(args.valid_data_path)

    if args.sanity_check:
        train_records = train_records[:2]
        valid_records = valid_records[:2]

    if args.overfit:
        print_main(
            f'[OVERFIT MODE] Using {args.overfit_samples} records for overfitting test'
        )
        train_records = train_records[: args.overfit_samples]
        valid_records = train_records
        args.epochs = 100
        args.eval_steps = 10
        args.logging_steps = 1
        args.save_steps = 50
        args.learning_rate = 1e-4

    def _has_pos_attr(r):
        n = len(r.news or [])
        return any(
            0 <= a.get('news_idx', -1) < n and float(a.get('score') or 0) > 0
            for a in (r.attributions or [])
        )

    if args.train_attributed_only:
        b_tr, b_va = len(train_records), len(valid_records)
        train_records = [r for r in train_records if _has_pos_attr(r)]
        valid_records = [r for r in valid_records if _has_pos_attr(r)]
        print_main(
            f'[attributed-only] train {b_tr}->{len(train_records)}, valid {b_va}->{len(valid_records)}'
        )

    if args.max_history_len is not None:
        for r in train_records + valid_records:
            if r.history and len(r.history) > args.max_history_len:
                r.history = r.history[-args.max_history_len :]
        print_main(
            f'[max-history-len] trimmed history to last {args.max_history_len} day(s)'
        )

    train_with_attr = sum(1 for r in train_records if r.attributions)
    valid_with_attr = sum(1 for r in valid_records if r.attributions)
    print_main(
        f'Train: {train_with_attr}/{len(train_records)} records have attributions'
    )
    print_main(
        f'Valid: {valid_with_attr}/{len(valid_records)} records have attributions'
    )

    if train_with_attr == 0:
        raise ValueError('No training records have attributions.')

    print_main('Full fine-tuning mode (FSDP)')

    world_model = MultiEventWorldModel(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        gradient_checkpointing=args.gradient_checkpointing,
        max_news=args.max_news,
        head_lr_multiplier=args.head_lr_multiplier,
        pooling_method=args.pooling_method,
        null_subsample_ratio=args.null_subsample_ratio,
        predict_delta=args.predict_delta,
        null_rho0=args.null_rho0,
        odds_eps=args.odds_eps,
        odds_temp=args.odds_temp,
    )

    # HF constraint: load_best_model_at_end=True requires save_steps to be
    # a round multiple of eval_steps. (Skipped when mid-checkpoints are off.)
    if (
        not args.no_mid_checkpoints
        and args.eval_steps > 0
        and args.save_steps % args.eval_steps != 0
    ):
        print_main(
            f'Adjusting save_steps from {args.save_steps} to {args.eval_steps} '
            'to satisfy load_best_model_at_end requirement.'
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
            fsdp_config['min_num_params'] = int(
                args.fsdp_min_num_params
            )  # size-based: wraps embedding too
        elif args.fsdp_transformer_layer_cls:
            fsdp_config['transformer_layer_cls_to_wrap'] = [
                args.fsdp_transformer_layer_cls
            ]
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
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        eval_steps=args.eval_steps,
        eval_strategy='steps',
        save_strategy='no' if args.no_mid_checkpoints else 'steps',
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
        report_to='none',
        run_name=f"{args.model_name.replace('/', '_')}_{args.output_dir.split('/')[-1]}",
        # DDP settings
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        local_rank=local_rank if world_size > 1 else -1,
    )

    best_checkpoint = world_model.train(
        train_records=train_records,
        valid_records=valid_records,
        training_args=training_args,
    )

    print_main(f'Best model saved to: {best_checkpoint}')

    # Only save on main process
    if is_main_process():
        world_model.save(best_checkpoint)

    # Cleanup distributed
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
