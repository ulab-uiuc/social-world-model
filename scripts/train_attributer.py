"""
Train PriorAttributer using KL divergence from PosteriorAttributer.

The input data should have attributions precomputed using precompute_attributions.py
(which uses PosteriorAttributer). The PosteriorAttributer derives the target
distribution with access to the realized future price; the PriorAttributer learns
to reproduce that distribution from the news + history + question alone (i.e.
without seeing the future outcome).

Single GPU:
    python train_attributer.py \
        --train-data-path ../data/attributed/train.jsonl \
        --valid-data-path ../data/attributed/valid.jsonl \
        --output-dir ../saves/prior_attributer

Multi-GPU (DDP):
    torchrun --nproc_per_node=4 train_attributer.py \
        --train-data-path ../data/attributed/train.jsonl \
        --valid-data-path ../data/attributed/valid.jsonl \
        --output-dir ../saves/prior_attributer
"""

import argparse
import os
from typing import Any, Dict

import torch
import torch.distributed as dist
from transformers import TrainingArguments

from swm.attributer import BasicPriorAttributer
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
        description='Train PriorAttributer using precomputed posterior attributions'
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
        help='Keep at most N checkpoints (best + most-recent). Prevents disk blowup for full-FT.',
    )
    parser.add_argument(
        '--ddp-timeout',
        type=int,
        default=1800,
        help='Distributed collective timeout (seconds). Raise well above the time a slow '
        'NFS checkpoint write takes, so other ranks waiting at the next allreduce do '
        'not trip the NCCL watchdog and abort the run mid-save.',
    )
    parser.add_argument(
        '--save-only-model',
        action='store_true',
        help='Save ONLY model weights, not optimizer/scheduler/rng. For FSDP full-FT this '
        'skips the huge fp32 optimizer-state gather (~2x params) — checkpoints shrink '
        'from ~3x to ~1x model size and save many minutes faster. No mid-run resume.',
    )
    parser.add_argument('--eval-steps', type=int, default=500)
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
        default=50,
        help='Max news articles per record (default: 50)',
    )
    parser.add_argument(
        '--target-temperature',
        type=float,
        default=0.5,
        help='Softmax temperature (shared train+inference; does NOT change '
        'output sharpness relative to target). Default 0.5.',
    )
    parser.add_argument(
        '--target-mode',
        type=str,
        default='normalize',
        choices=['normalize', 'odds'],
        help="Target dist over (news∪no-news). 'odds': score->odds, null gets raw mass "
        '--null-odds, then normalize (weak scores->high no-news, null emerges).',
    )
    parser.add_argument(
        '--null-odds',
        type=float,
        default=1.0,
        help='Raw odds mass rho_0 for the no-news slot in --target-mode odds. Higher = more conservative (more null mass).',
    )
    parser.add_argument('--odds-eps', type=float, default=1e-3)
    parser.add_argument(
        '--odds-temp',
        type=float,
        default=1.0,
        help='Smoothing T for odds**(1/T); T>1 flattens the target.',
    )
    parser.add_argument(
        '--target-sharpen',
        type=float,
        default=1.0,
        help='Sharpen the KL target: p_dist ∝ score**this. >1 makes the '
        'attributer more decisive (top news gets more mass). Default 1.0.',
    )
    parser.add_argument(
        '--routing-loss-weight',
        type=float,
        default=0.0,
        help='Weight of the routing BCE: pushes the no-news prob -> 1 for '
        'null records, 0 for has-news, directly training the '
        '"is there causal news?" classifier. Default 0 (off).',
    )
    parser.add_argument(
        '--neg-bce-weight',
        type=float,
        default=0.0,
        help='Per-news relevance BCE weight: pushes sigmoid(logit)->1 for gold-attributed news, 0 otherwise. Supplies forward-KL the negative-suppression it lacks.',
    )
    parser.add_argument(
        '--reverse-kl',
        action='store_true',
        help='Use reverse KL(q||p) (mode-seeking: sharpens the distribution and '
        'actively suppresses off-target/irrelevant news) instead of forward KL(p||q).',
    )
    parser.add_argument(
        '--head-lr-multiplier',
        type=float,
        default=1.0,
        help='LR multiplier for the regression head. >1 fixes head under-fitting that keeps the attributer output flatter (eff~5) than the KL target (eff~2).',
    )
    parser.add_argument(
        '--per-news-bce',
        action='store_true',
        help='Per-news Bernoulli mode: drop the softmax/KL; train each news as an '
        'independent sigmoid with soft-target BCE to its 0-1 posterior relevance '
        '(matches the 235B per-news labels). no-news is emergent = Π(1-p_i). '
        'Decouples ranking from routing — no softmax competition.',
    )
    parser.add_argument(
        '--null-subsample-ratio',
        type=float,
        default=1.0,
        help='Fraction of null records (no positive attributions) to keep in TRAIN dataset. '
        '1.0=keep all (default), <1.0 dilutes null supervision. Valid set always keeps ratio=1.0.',
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='social-world-model',
        help='Wandb project name (default: social-world-model)',
    )
    parser.add_argument(
        '--resume-from-checkpoint',
        type=str,
        default=None,
        nargs='?',
        const='True',
        help='Resume from checkpoint. Pass path or use flag alone for auto-detect.',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    set_seed(args.seed)

    # Initialize wandb project (only on main process)
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

    attributer = BasicPriorAttributer(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        gradient_checkpointing=args.gradient_checkpointing,
        max_news=args.max_news,
        target_temperature=args.target_temperature,
        null_subsample_ratio=args.null_subsample_ratio,
        target_mode=args.target_mode,
        null_odds=args.null_odds,
        odds_eps=args.odds_eps,
        odds_temp=args.odds_temp,
        target_sharpen=args.target_sharpen,
        routing_loss_weight=args.routing_loss_weight,
        reverse_kl=args.reverse_kl,
        neg_bce_weight=args.neg_bce_weight,
        per_news_bce=args.per_news_bce,
        head_lr_multiplier=args.head_lr_multiplier,
    )

    # HF constraint: load_best_model_at_end=True requires save_steps to be
    # a round multiple of eval_steps.
    if args.eval_steps > 0 and args.save_steps % args.eval_steps != 0:
        print_main(
            f'Adjusting save_steps from {args.save_steps} to {args.eval_steps} '
            'to satisfy load_best_model_at_end requirement.'
        )
        args.save_steps = args.eval_steps

    fsdp_kwargs: Dict[str, Any] = {}
    if args.fsdp:
        fsdp_kwargs['fsdp'] = args.fsdp
        # FULL_STATE_DICT so the root FSDP unit (embeddings/final norm) is gathered
        # on save; otherwise it's written as an unloadable 1-D flat shard.
        fsdp_config: Dict[str, Any] = {'state_dict_type': 'FULL_STATE_DICT'}
        if args.fsdp_transformer_layer_cls:
            fsdp_config['transformer_layer_cls_to_wrap'] = [
                args.fsdp_transformer_layer_cls
            ]
        fsdp_kwargs['fsdp_config'] = fsdp_config

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
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        ddp_timeout=args.ddp_timeout,
        eval_steps=args.eval_steps,
        eval_strategy='steps',
        save_strategy='steps',
        fp16=args.fp16,
        bf16=args.bf16,
        **fsdp_kwargs,
        metric_for_best_model='eval_loss',
        greater_is_better=False,  # Lower loss is better
        # load_best_model_at_end=True would torch.load at training end, which
        # tripped HF's CVE-2025-32434 check on torch<2.6. Skip the auto-reload;
        # `trainer.state.best_model_checkpoint` is still populated.
        load_best_model_at_end=False,
        save_safetensors=True,
        remove_unused_columns=False,
        report_to='wandb' if is_main_process() else 'none',
        logging_first_step=True,
        # DDP settings
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        local_rank=local_rank if world_size > 1 else -1,
    )

    # Train using precomputed attributions (no need to pass posterior_attributer)
    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt == 'True':
        resume_ckpt = True  # Auto-detect latest checkpoint
    best_checkpoint = attributer.train(
        train_records=train_records,
        valid_records=valid_records,
        training_args=training_args,
        resume_from_checkpoint=resume_ckpt,
    )

    print_main(f'Best model saved to: {best_checkpoint}')

    # Only save on main process
    if is_main_process():
        attributer.save(best_checkpoint)

    # Cleanup distributed
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
