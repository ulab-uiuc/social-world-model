import argparse
import json
import os

from swm.utils.reasoner import PolyMarketDailyNewsReasoner
from swm.utils.utils import load_dailynews_data, load_polymarket_data


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze PolyMarket data with OpenAI')

    parser.add_argument(
        '--news-path',
        type=str,
        default='../data/processed_dailynews/dailynews_data_processed.jsonl',
    )
    parser.add_argument(
        '--market-path',
        type=str,
        default='../data/splitted_polymarket/polymarket_data_processed_Crypto_test.jsonl',
    )
    parser.add_argument('--analysis-date', type=str, default='2024-12-01')
    parser.add_argument('--top-k', type=int, default=2)
    parser.add_argument('--news-window', type=int, default=1)
    parser.add_argument('--output-path', type=str, default='analysis_results.json')

    return parser.parse_args()


def main():
    args = parse_args()
    openai_api_key = os.environ.get('OPENAI_API_KEY')

    news = load_dailynews_data(args.news_path)
    markets = load_polymarket_data(args.market_path)

    reasoner = PolyMarketDailyNewsReasoner(
        openai_api_key=openai_api_key,
        corpus_markets=markets,
        corpus_news=news,
        top_k=args.top_k,
        news_window_days=args.news_window,
    )

    try:
        scores = reasoner.analyze(args.analysis_date)
        with open(args.output_path, 'w') as f:
            json.dump(scores, f, indent=4)
        print(f'Results saved to {args.output_path}')
    except Exception as e:
        print(f'An error occurred during analysis: {e}')
        scores = []


if __name__ == '__main__':
    main()
