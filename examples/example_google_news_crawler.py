#!/usr/bin/env python3
"""Google News crawler example."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swm.utils.crawler import GoogleNewsCrawler


def main():
    crawler = GoogleNewsCrawler(min_delay=2.0, max_delay=4.0)

    crawler.crawl(
        query='artificial intelligence',
        start_date='2024-11-01',
        end_date='2024-11-29',
        output_file='../data/google_news.jsonl',
        max_pages=5,
        fetch_full_content=True,  # Fetch full article content (slower but more text)
    )


if __name__ == '__main__':
    main()
