from swm.utils.crawler import PolyMarketEventCollector, PolyMarketHistoryCollector


def main():
    event_file = '../data/polymarket_event_data.jsonl'
    event_with_history_file = '../data/polymarket_event_with_history_data.jsonl'

    event_collector = PolyMarketEventCollector(output_file=event_file)
    event_collector.collect_event(start_offset=0, end_offset=100)

    print(f'\nEvents saved to {event_file}.')

    history_collector = PolyMarketHistoryCollector(
        input_file=event_file, output_file=event_with_history_file
    )
    history_collector.collect_history()

    print(f'\nMarket histories saved to {event_with_history_file}.')


if __name__ == '__main__':
    main()
