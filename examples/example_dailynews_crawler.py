import os
from datetime import datetime, timedelta

from swm.utils.crawler import DailyNewsCrawler


def main():
    # Configuration
    date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')  # Yesterday's date
    output_file = f'./{date}.json'
    api_token = os.environ.get('NEWS_API_KEY')

    try:
        crawler = DailyNewsCrawler(
            input_date=date,
            output_file=output_file,
            api_token=api_token,
        )

        crawler.crawl()

    except Exception as e:
        print(f'Error in main: {e}', exc_info=True)


if __name__ == '__main__':
    main()
