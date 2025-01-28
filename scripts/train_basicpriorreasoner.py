import argparse

from peft import LoraConfig
from transformers import TrainingArguments

# Example imports – adapt to your actual module locations
from swm.reasoner import BasicPriorReasoner
from swm.utils.posterior_reasoner import BasicPosteriorReasoner
from swm.utils.utils import load_dailynews_data, load_polymarket_data, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train the Basic Prior Reasoner')

    parser.add_argument('--train-data-path', type=str, required=True)
    parser.add_argument('--valid-data-path', type=str, required=True)
    parser.add_argument('--corpus-news-path', type=str, required=True)
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--cache-dir', type=str, default='./cache')
    parser.add_argument('--output-dir', type=str, default='./output')
    parser.add_argument('--eval-batch-size', type=int, default=8)
    parser.add_argument('--max-seq-length', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-batch-size', type=int, default=8)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-steps', type=int, default=0)
    parser.add_argument('--max-grad-norm', type=float, default=1.0)
    parser.add_argument('--logging-steps', type=int, default=100)
    parser.add_argument('--save-steps', type=int, default=500)
    parser.add_argument('--eval-steps', type=int, default=500)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--lora-alpha', type=float, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.1)
    parser.add_argument('--r', type=int, default=16)
    parser.add_argument('--reasoner-name', type=str, default='gpt-4')
    parser.add_argument('--reasoner-max-news-items', type=int, default=10)
    parser.add_argument('--sanity-check', action='store_true')
    parser.add_argument('--reasoner-cache-dir', type=str, default='./cache')

    return parser.parse_args()


def train(args):
    # Set random seed for reproducibility
    set_seed(args.seed)

    # Possibly load a subset of data if doing a quick sanity check
    if args.sanity_check:
        train_data = load_polymarket_data(args.train_data_path)[:2]
        valid_data = load_polymarket_data(args.valid_data_path)[:2]
        corpus_news = load_dailynews_data(args.corpus_news_path)
    else:
        train_data = load_polymarket_data(args.train_data_path)
        valid_data = load_polymarket_data(args.valid_data_path)
        corpus_news = load_dailynews_data(args.corpus_news_path)

    # Prepare LoRA configuration if your BasicPriorReasoner supports it
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=['q_proj', 'v_proj'],  # adapt for your model
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
    )

    # Initialize the BasicPriorReasoner
    prior_reasoner = BasicPriorReasoner(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        max_seq_length=args.max_seq_length,
        lora_config=lora_config,
    )

    # Prepare standard Hugging Face training args
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
    )

    # If your Prior reasoner depends on a Posterior reasoner (for generating p-dist),
    # set it up here:
    posterior_reasoner = BasicPosteriorReasoner(
        model_name=args.reasoner_name,
        max_news_items=args.reasoner_max_news_items,
        corpus_news=corpus_news,
        cache_dir=args.reasoner_cache_dir,
    )

    # Train the Prior reasoner using some method like prior_reasoner.train_with_kl(...)
    best_model_checkpoint = prior_reasoner.train(
        train_data=train_data,
        valid_data=valid_data,
        training_args=training_args,
        posterior_reasoner=posterior_reasoner,
    )

    # Save the final or best model
    prior_reasoner.save(best_model_checkpoint)


if __name__ == '__main__':
    args = parse_args()
    train(args)
