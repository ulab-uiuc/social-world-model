#!/usr/bin/env python3
"""Kalshi crawler example using private key authentication."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swm.utils.crawler import KalshiCrawler


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    private_key_path = os.path.join(script_dir, 'swm.txt')
    
    if os.getenv('KALSHI_PRIVATE_KEY_PATH'):
        private_key_path = os.getenv('KALSHI_PRIVATE_KEY_PATH')
    
    if not os.path.exists(private_key_path):
        print(f"❌ Private key not found: {private_key_path}")
        return
    
    
    crawler = KalshiCrawler(
        output_file='../data/raw_kalshi/kalshi_data_raw.jsonl',
        private_key_path=private_key_path,
    )
    
    crawler.collect_markets(max_markets=5000000)
    
    print(f"\n📂 Data saved.")


if __name__ == "__main__":
    main()
