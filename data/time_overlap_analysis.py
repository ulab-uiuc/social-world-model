import json
from typing import List, Dict, Tuple
from collections import defaultdict


def load_markets(file_path: str) -> List[Dict]:
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def extract_sequence_to_tags(
    markets: List[Dict], window_size: int = 5
) -> Dict[Tuple[int, ...], List[List[str]]]:
    """
    返回 dict: {sequence_tuple: [tags1, tags2, ...]}
    每个 tags 是该序列对应 market 的 tags
    """
    seq_to_tags = defaultdict(list)
    for market in markets:
        tags = market.get("categories", [])
        series = market.get("daily_time_series", {}).get("Yes", [])
        if len(series) <= window_size:
            continue

        for start_idx in range(len(series) - window_size):
            window = series[start_idx : start_idx + window_size]
            target = series[start_idx + window_size]
            if target["p"] - window[-1]["p"] >= 0.25:
                full_sequence = window + [target]
                sequence_tuple = tuple(item["t"] for item in full_sequence)
                seq_to_tags[sequence_tuple].append(tags)
    return seq_to_tags


def main():
    train_path = "swm_bench_train.jsonl"
    test_path = "swm_bench_test.jsonl"

    train_data = load_markets(train_path)
    test_data = load_markets(test_path)

    train_seq_tags = extract_sequence_to_tags(train_data)
    test_seq_tags = extract_sequence_to_tags(test_data)

    shared_sequences = set(train_seq_tags.keys()) & set(test_seq_tags.keys())

    print(f" {len(shared_sequences)} 个时间序列在 train 和 test 中都出现过")

    for seq in sorted(
        shared_sequences, key=lambda s: -len(train_seq_tags[s]) - len(test_seq_tags[s])
    ):
        train_tags = train_seq_tags[seq]
        test_tags = test_seq_tags[seq]
        total_occurrences = len(train_tags) + len(test_tags)

        if total_occurrences > 1:  # 至少重复一次才打印
            print(f"\n🕒 Sequence: {seq}")
            print(f"  - Train Tags ({len(train_tags)}): {train_tags}")
            print(f"  - Test Tags  ({len(test_tags)}): {test_tags}")


if __name__ == "__main__":
    main()
