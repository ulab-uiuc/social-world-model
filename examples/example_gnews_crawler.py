#!/usr/bin/env python3
"""GNews API crawler example.

GNews API: https://gnews.io/
- Free tier: 100 requests/day, 10 articles per request
- Supports date filtering
- More stable than scraping Google News

Usage:
    export GNEWS_API_KEY="your-api-key"
    python example_gnews_crawler.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swm.utils.crawler import GNewsCrawler


def main():
    # Get API key from environment
    api_key = os.environ.get("GNEWS_API_KEY")
    if not api_key:
        print("Please set GNEWS_API_KEY environment variable")
        print("Get your free API key at: https://gnews.io/")
        return

    crawler = GNewsCrawler(api_key=api_key)

    crawler.crawl(
        query="Bitcoin",
        start_date="2025-11-01",
        end_date="2025-11-10",
        output_file="../data/gnews_test.jsonl",
        max_results=10,
        lang="en",
        country="us",
    )


if __name__ == "__main__":
    main()

