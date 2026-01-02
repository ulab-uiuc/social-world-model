from swm.utils.crawler import PolyMarketEventCrawler, PolyMarketHistoryCrawler


def main():
    event_file = '../data/raw_polymarket/polymarket_event_data.jsonl'
    event_with_history_file = (
        '../data/raw_polymarket/polymarket_event_with_history_data.jsonl'
    )

    event_collector = PolyMarketEventCrawler(output_file=event_file)
    event_collector.collect(start_offset=0, end_offset=20000)

    print(f'\nEvents saved to {event_file}.')

    history_collector = PolyMarketHistoryCrawler(
        input_file=event_file, output_file=event_with_history_file
    )
    history_collector.collect()

    print(f'\nMarket histories saved to {event_with_history_file}.')


if __name__ == '__main__':
    main()
