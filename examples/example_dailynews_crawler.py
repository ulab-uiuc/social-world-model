import argparse
import os

from swm.utils.crawler import DailyNewsCrawler


def parse_args():
    parser = argparse.ArgumentParser(
        description='Crawl daily news articles for a date range'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2024-01-01',
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='2025-01-02',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../data/raw_news',
    )
    parser.add_argument(
        '--api-key',
        type=str,
    )

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    output_file = os.path.join(
        args.output_dir, f'daily_news_{args.start_date}_{args.end_date}.jsonl'
    )

    api_token = args.api_key or os.environ.get('NEWS_API_KEY')

    crawler = DailyNewsCrawler(
        start_date=args.start_date,
        end_date=args.end_date,
        output_file=output_file,
        api_token=api_token,
    )
    crawler.crawl()


if __name__ == '__main__':
    main()
