import json
from typing import List, Dict, Tuple, Set


def load_markets(file_path: str) -> List[Dict]:
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def extract_valid_sequences(
    markets: List[Dict], window_size: int = 5
) -> Set[Tuple[Tuple[int, float], ...]]:
    all_sequences = set()
    for market in markets:
        series = market.get("daily_time_series", {}).get("Yes", [])
        if len(series) <= window_size:
            continue

        for start_idx in range(len(series) - window_size):
            window = series[start_idx : start_idx + window_size]
            target = series[start_idx + window_size]
            if target["p"] - window[-1]["p"] >= 0.25:
                full_sequence = window + [target]
                sequence_tuple = tuple(item["t"] for item in full_sequence)
                all_sequences.add(sequence_tuple)
    return all_sequences


def main():
    train_path = "data/swm_bench_train.jsonl"
    test_path = "data/swm_bench_test.jsonl"

    train_data = load_markets(train_path)
    test_data = load_markets(test_path)

    train_seq_set = extract_valid_sequences(train_data)
    test_seq_set = extract_valid_sequences(test_data)

    overlap = train_seq_set & test_seq_set

    print(f"length of train set: {len(train_seq_set)}")
    print(f"length of test set: {len(test_seq_set)}")
    print(f"exactly same time points：{len(overlap)}")
    print(f"ratio：{len(overlap) / len(test_seq_set):.3%}")


if __name__ == "__main__":
    main()
