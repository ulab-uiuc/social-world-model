import argparse
from pathlib import Path

import pandas as pd
from peft import LoraConfig

# Adapt import paths to your actual code structure
from swm.reasoner import BasicPriorReasoner
from swm.utils.metric import calculate_kl_divergence
from swm.utils.posterior_reasoner import BasicPosteriorReasoner
from swm.utils.utils import load_dailynews_data, load_polymarket_data, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Test BasicPriorReasoner')

    # Data paths
    parser.add_argument('--test-data-path', type=str, required=True)
    parser.add_argument('--corpus-news-path', type=str, required=True)
    parser.add_argument('--model-checkpoint', type=str, required=True)

    # Model configs
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--max-seq-length', type=int, default=1024)
    parser.add_argument('--cache-dir', type=str, default='./cache')
    parser.add_argument('--output-dir', type=str, default='./output')

    # Test configs
    parser.add_argument('--test-batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')

    # LoRA configs
    parser.add_argument('--lora-alpha', type=float, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.1)
    parser.add_argument('--r', type=int, default=16)

    # Reasoner configs
    parser.add_argument('--reasoner-name', type=str, default='gpt-4')
    parser.add_argument('--reasoner-max-news-items', type=int, default=10)

    # Debug
    parser.add_argument('--sanity-check', action='store_true')

    return parser.parse_args()


def test_prior_reasoner(args):
    # Set random seed
    set_seed(args.seed)

    # Load data
    if args.sanity_check:
        test_data = load_polymarket_data(args.test_data_path)[:2]
        corpus_news = load_dailynews_data(args.corpus_news_path)
    else:
        test_data = load_polymarket_data(args.test_data_path)
        corpus_news = load_dailynews_data(args.corpus_news_path)

    # Setup LoRA config if your prior reasoner supports it
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=['q_proj', 'v_proj'],  # adapt for your model
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type='CAUSAL_LM',
    )

    # Initialize prior reasoner
    prior_reasoner = BasicPriorReasoner(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        max_seq_length=args.max_seq_length,
        lora_config=lora_config,
    )

    # Load trained checkpoint
    prior_reasoner.load(args.model_checkpoint)

    # If your prior reasoner needs a posterior reasoner at inference:
    posterior_reasoner = BasicPosteriorReasoner(
        model_name=args.reasoner_name,
        max_news_items=args.reasoner_max_news_items,
        corpus_news=corpus_news,
    )

    # Run predictions
    # Depending on your prior reasoner’s `predict` signature,
    # you might do something like:
    results = prior_reasoner.predict(
        markets=test_data,
        posterior_reasoner=posterior_reasoner,
        batch_size=args.test_batch_size,
    )

    # Extract predictions & ground truths
    predictions = [r['q_dist'] for r in results]
    ground_truth = [r['p_dist'] for r in results]

    kl_divs = []
    for pred, gth in zip(predictions, ground_truth):
        kl_div = calculate_kl_divergence(pred, gth)['kl_div']
        kl_divs.append(kl_div)
    metrics = {'kl_div': sum(kl_divs) / len(kl_divs)}

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save metrics
    pd.DataFrame([metrics]).to_csv(output_dir / 'metrics.csv', index=False)

    # Save predictions
    pd.DataFrame(results).to_csv(output_dir / 'predictions.csv', index=False)

    # Print metrics to console
    print('Test Metrics:')
    for metric, value in metrics.items():
        print(f'{metric}: {value:.4f}')


if __name__ == '__main__':
    args = parse_args()
    test_prior_reasoner(args)
