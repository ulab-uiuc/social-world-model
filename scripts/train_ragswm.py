import argparse
from pathlib import Path
from typing import List

import jsonlines
import torch
from peft import LoraConfig
from transformers import TrainingArguments

from swm.data import PolyMarketData
from swm.swm import RAGSocialWM


def load_polymarket_data(data_path: str) -> List[PolyMarketData]:
    with jsonlines.open(data_path, 'r') as reader:
        data = list(reader)
    return [PolyMarketData.from_dict(d) for d in data]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and evaluate the RAG Social Wisdom Model'
    )

    # Data paths
    parser.add_argument(
        '--train-data-path',
        type=str,
        default='../data/splitted_polymarket/polymarket_data_processed_Crypto_train.jsonl',
    )
    parser.add_argument(
        '--valid-data-path',
        type=str,
        default='../data/splitted_polymarket/polymarket_data_processed_Crypto_dev.jsonl',
    )
    parser.add_argument(
        '--test-data-path',
        type=str,
        default='../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl',
    )
    parser.add_argument(
        '--corpus-data-path',
        type=str,
        default='../data/splitted_polymarket/polymarket_data_processed_Crypto_train.jsonl',
    )
    parser.add_argument('--cache-dir', type=str, default='./cache')

    # Model parameters
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--retriever-name', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--fp16', action='store_true')

    # LoRA parameters
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.1)
    parser.add_argument(
        '--target-modules', type=str, nargs='+', default=['q_proj', 'v_proj']
    )

    # Training parameters
    parser.add_argument('--train-batch-size', type=int, default=16)
    parser.add_argument('--eval-batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--grad-accum-steps', type=int, default=1)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--logging-steps', type=int, default=10)
    parser.add_argument('--save-steps', type=int, default=500)
    parser.add_argument('--eval-steps', type=int, default=500)

    # Retriever parameters
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--retriever-batch-size', type=int, default=32)
    parser.add_argument('--max-seq-length', type=int, default=1024)

    # Output paths
    parser.add_argument('--output-dir', type=str, default='../saves_crypto')
    parser.add_argument('--predictions-path', type=str, default='predictions.csv')
    parser.add_argument('--model-save-path', type=str, default='model')

    return parser.parse_args()


def train(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    train_data = load_polymarket_data(args.train_data_path)
    valid_data = load_polymarket_data(args.valid_data_path)
    corpus_data = load_polymarket_data(args.corpus_data_path)

    # Create LoraConfig based on parsed arguments
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias='none',
        task_type='CAUSAL_LM',
    )

    # Detect device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')

    # Initialize RAGSocialWM with lora_config and device
    model = RAGSocialWM(
        model_name=args.model_name,
        retriever_name=args.retriever_name,
        cache_dir=args.cache_dir,
        lora_config=lora_config,
        corpus_markets=corpus_data,
        max_seq_length=args.max_seq_length,
        top_k=args.top_k,
        retriever_batch_size=args.retriever_batch_size,
    )

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy='steps',
        save_strategy='steps',
        fp16=args.fp16,
        load_best_model_at_end=True,
        metric_for_best_model='mse',
        greater_is_better=False,
    )

    # Train the model, passing the device
    best_model_checkpoint = model.train(
        train_data=train_data,
        valid_data=valid_data,
        training_args=training_args,
        device=device,
    )
    print(f'Best model checkpoint saved at: {best_model_checkpoint}')


if __name__ == '__main__':
    args = parse_args()
    train(args)
