import os
from datetime import datetime, timedelta

from swm.utils.crawler import DailyNewsCrawler


def main():
    date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    output_file = 'crypto_news_2024_01_20.json'
    api_key = os.getenv('NEWS_API_KEY')

    crawler = DailyNewsCrawler(
        input_date=date, output_file=output_file, api_key=api_key, keywords=['bitcoin']
    )

    crawler.crawl()


if __name__ == '__main__':
    main()
