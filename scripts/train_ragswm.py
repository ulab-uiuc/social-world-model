# train_ragswm.py

import argparse

import torch
from peft import LoraConfig
from transformers import TrainingArguments

from swm.data import PolyMarketData
from swm.swm import RAGSocialWM


def set_seed(seed: int = 42):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_polymarket_data(data_path):
    import jsonlines

    with jsonlines.open(data_path, 'r') as reader:
        return [PolyMarketData.from_dict(d) for d in reader]


def parse_args():
    parser = argparse.ArgumentParser(description='Train the RAG Social Wisdom Model')

    parser.add_argument('--train-data-path', type=str, required=True)
    parser.add_argument('--valid-data-path', type=str, required=True)
    parser.add_argument('--corpus-data-path', type=str, required=True)
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--retriever-name', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--eval-batch-size', type=int, default=8)
    parser.add_argument('--cache-dir', type=str, default='./cache')
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--predictions-path', type=str, default='predictions.csv')
    parser.add_argument('--max-seq-length', type=int, default=1024)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--retriever-batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-batch-size', type=int, default=8)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-steps', type=int, default=0)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--logging-steps', type=int, default=100)
    parser.add_argument('--save-steps', type=int, default=100)
    parser.add_argument('--eval-steps', type=int, default=100)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--lora-alpha', type=float, default=0.5)
    parser.add_argument('--lora-dropout', type=float, default=0.1)
    parser.add_argument('--r', type=int, default=1)
    return parser.parse_args()


def train(args):
    set_seed(args.seed)
    train_data = load_polymarket_data(args.train_data_path)
    valid_data = load_polymarket_data(args.valid_data_path)
    corpus_data = load_polymarket_data(args.corpus_data_path)

    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
    )

    rag_swm = RAGSocialWM(
        model_name=args.model_name,
        retriever_name=args.retriever_name,
        cache_dir=args.cache_dir,
        lora_config=lora_config,
        corpus_markets=corpus_data,
        max_seq_length=args.max_seq_length,
        top_k=args.top_k,
        retriever_batch_size=args.retriever_batch_size,
    )

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
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        save_safetensors=False,
    )

    best_model_checkpoint = rag_swm.train(
        train_data=train_data,
        valid_data=valid_data,
        training_args=training_args,
    )

    rag_swm.save(best_model_checkpoint)


if __name__ == '__main__':
    args = parse_args()
    train(args)
