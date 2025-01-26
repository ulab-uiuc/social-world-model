import argparse
import os

import jsonlines

from swm.reasoner import BasicPosteriorReasoner
from swm.utils.utils import load_dailynews_data, load_polymarket_data
from tqdm import tqdm

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
    parser.add_argument('--date', type=str, default='2024-08-05')
    parser.add_argument('--max-news', type=int, default=30)
    parser.add_argument('--output-path', type=str, default='analysis_results.jsonl')
    parser.add_argument('--model-name', type=str, default='gpt-4o')
    return parser.parse_args()

def main():
    args = parse_args()
    
    news = load_dailynews_data(args.news_path)
    markets = load_polymarket_data(args.market_path)
    
    reasoner = BasicPosteriorReasoner(
        corpus_news=news,
        model_name=args.model_name,
        max_news_items=args.max_news
    )
    
    results = []
    for market in tqdm(markets):
        results = reasoner.reason(args.date, market)
        if results:
            results.append({
                'market_id': market.market_id,
                'question': market.question,
                'results': results
            })
    
    with jsonlines.open(args.output_path, mode='w') as writer:
        for result in results:
            writer.write(result)
    
    print(f'Analyzed {len(results)} markets. Results saved to {args.output_path}')

if __name__ == '__main__':
    main()
