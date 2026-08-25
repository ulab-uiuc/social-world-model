#!/usr/bin/env python3
"""Fine-tune the regression head on streaming contexts.

Same architecture as the shipped checkpoint -- backbone plus a regression head
read off the last token -- and the same target, the 24h delta off
`history[-1].p`. The only thing that changes is what the model gets to read: one
long prompt holding the market's own past reactions instead of 1024 tokens
holding one headline.

That is deliberate. Keeping the head and the target fixed makes the comparison
against `swmbench/swm-wm-jin10-daily-7b` a clean single-variable one, and lets
the same eval and the same backtest score both.

Memory: a 7B full fine-tune under AdamW needs roughly 100GB of parameter and
optimizer state alone, so this wants FSDP across several GPUs plus gradient
checkpointing. On 96GB cards, four ranks is comfortable at 30k tokens.

    torchrun --nproc_per_node=4 scripts/train_streaming_sft.py \\
        --data-dir data/streaming --init-from <ckpt> \\
        --model-name Qwen/Qwen2.5-7B-Instruct \\
        --output-dir saves/streaming_sft --max-seq-length 30000 \\
        --bf16 --gradient-checkpointing --fsdp "full_shard auto_wrap" \\
        --fsdp-transformer-layer-cls Qwen2DecoderLayer
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, TrainingArguments

from swm.streaming import (
    StreamingRegressionDataset,
    build_streaming_trainer_class,
    collate_streaming,
)
from swm.utils.regressor import LLMRegressor, LLMRegressorConfig


def is_main() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def log(*args):
    if is_main():
        print(*args, flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True, help='from build_streaming_data.py')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--model-name', default='Qwen/Qwen2.5-7B-Instruct',
                   help='tokenizer, and the backbone when --init-from is unset')
    p.add_argument('--init-from', default=None,
                   help='an existing LLMRegressor checkpoint to continue from. '
                        'Continuing from the shipped world model asks "does the '
                        'streaming context add anything to what we already have"; '
                        'omitting it trains the head from scratch, which on 3k '
                        'examples confounds that question with head warm-up.')
    p.add_argument('--max-seq-length', type=int, default=30000)
    p.add_argument('--epochs', type=float, default=2.0)
    p.add_argument('--train-batch-size', type=int, default=1)
    p.add_argument('--eval-batch-size', type=int, default=1)
    p.add_argument('--gradient-accumulation-steps', type=int, default=8)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--head-lr-multiplier', type=float, default=10.0)
    p.add_argument('--weight-decay', type=float, default=0.01)
    p.add_argument('--warmup-ratio', type=float, default=0.03)
    p.add_argument('--max-grad-norm', type=float, default=1.0)
    p.add_argument('--logging-steps', type=int, default=10)
    p.add_argument('--eval-steps', type=int, default=100)
    p.add_argument('--save-steps', type=int, default=100)
    p.add_argument('--bf16', action='store_true')
    p.add_argument('--gradient-checkpointing', action='store_true')
    p.add_argument('--fsdp', default='')
    p.add_argument('--fsdp-transformer-layer-cls', default='Qwen2DecoderLayer')
    p.add_argument('--limit-train', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def load_rows(path: Path, limit=None):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    return rows[:limit] if limit else rows


def build_model(args):
    if args.init_from:
        log(f'continuing from {args.init_from}')
        model = LLMRegressor.from_pretrained(args.init_from)
    else:
        log(f'fresh head on {args.model_name}')
        model = LLMRegressor(
            LLMRegressorConfig(
                base_model_name_or_path=args.model_name,
                max_length=args.max_seq_length,
                pooling_method='last_token',
                predict_delta=True,
            )
        )
    # The head is fp32 while the backbone loads bf16; FSDP refuses to flatten
    # mixed dtypes, so make the whole module fp32 and let TrainingArguments'
    # bf16 flag drive mixed precision -- the same dance world_model.py does.
    model = model.float()
    model.config.max_length = args.max_seq_length
    if args.gradient_checkpointing and hasattr(model.llm, 'gradient_checkpointing_enable'):
        model.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False}
        )
        log('gradient checkpointing on')
    return model


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    train_rows = load_rows(data_dir / 'train.jsonl', args.limit_train)
    valid_rows = load_rows(data_dir / 'valid.jsonl')
    log(f'train {len(train_rows)}  valid {len(valid_rows)}')

    train_ds = StreamingRegressionDataset(train_rows, tokenizer, args.max_seq_length)
    valid_ds = StreamingRegressionDataset(valid_rows, tokenizer, args.max_seq_length)

    model = build_model(args)

    fsdp_kwargs = {}
    if args.fsdp:
        fsdp_kwargs['fsdp'] = args.fsdp
        fsdp_kwargs['fsdp_config'] = {
            'state_dict_type': 'FULL_STATE_DICT',
            'transformer_layer_cls_to_wrap': [args.fsdp_transformer_layer_cls],
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        eval_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=False,
        bf16=args.bf16,
        gradient_checkpointing=False,  # enabled directly on the backbone above
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
        **fsdp_kwargs,
    )

    trainer_cls = build_streaming_trainer_class()
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=lambda batch: collate_streaming(batch, pad_id),
    )

    # A head learning faster than the backbone: the backbone is already tuned
    # for this task, the head has to re-fit a pooled vector drawn from a
    # 30k-token sequence rather than a 1k one.
    if args.head_lr_multiplier != 1.0:
        base_optimizer = trainer.create_optimizer

        def create_optimizer():
            optimizer = base_optimizer()
            for group in optimizer.param_groups:
                group.setdefault('lr', args.learning_rate)
            head_ids = {
                id(p) for n, p in model.named_parameters() if 'regression_head' in n
            }
            for group in optimizer.param_groups:
                if all(id(p) in head_ids for p in group['params']):
                    group['lr'] = args.learning_rate * args.head_lr_multiplier
            return optimizer

        trainer.create_optimizer = create_optimizer

    trainer.train()
    final = Path(args.output_dir) / 'final-model'
    trainer.save_model(str(final))
    log(f'saved -> {final}')


if __name__ == '__main__':
    main()
