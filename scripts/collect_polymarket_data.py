from swm.polymarket_price import collect_histories


def main():
    # Example usage
    events_file = "../data/data_with_offset_0112.jsonl"
    final_file = "../data/data_with_offset_0112_merged_with_history_test.jsonl"

    # Step 1: Collect events
    #print("Step 1: Collecting events...")
    #collect_events(0, 15000, events_file)

    # Step 2: Collect histories
    #print("\nStep 2: Collecting market histories...")
    collect_histories(events_file, final_file)

if __name__ == '__main__':
    main()
