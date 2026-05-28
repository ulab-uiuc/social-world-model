"""Drop records where |z_score| > 1 but the attributor returned no positive
attributions — these are likely attributor failures, not legitimate nulls.

Reads data/v6/*.jsonl and writes cleaned versions to data/v6/v6_clean/.
Also splits cleaned train into train + valid_subset150 (fixed seed).
"""
import argparse
import json
import random
from pathlib import Path


Z_THRESHOLD = 1.0
VALID_SIZE = 150
VALID_SEED = 42


def is_zero_attr(record: dict) -> bool:
    attrs = record.get('attributions') or []
    if not attrs:
        return True
    return all(float(a.get('score', 0) or 0) <= 0 for a in attrs)


def should_drop(record: dict) -> bool:
    z = record.get('z_score')
    if z is None:
        return False
    return abs(z) > Z_THRESHOLD and is_zero_attr(record)


def read_clean(src: Path) -> tuple[list[str], int]:
    """Return (kept_jsonl_lines, dropped_count)."""
    kept_lines = []
    dropped = 0
    with src.open() as fin:
        for line in fin:
            r = json.loads(line)
            if should_drop(r):
                dropped += 1
                continue
            kept_lines.append(line)
    return kept_lines, dropped


def write_lines(dst: Path, lines: list[str]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open('w') as fout:
        fout.writelines(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir', default='data/v6')
    parser.add_argument('--out-subdir', default='v6_clean')
    parser.add_argument(
        '--files',
        nargs='+',
        default=['train.jsonl', 'test_kalshi.jsonl', 'test_polymarket.jsonl'],
    )
    parser.add_argument('--valid-size', type=int, default=VALID_SIZE)
    parser.add_argument('--valid-seed', type=int, default=VALID_SEED)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    out_dir = src_dir / args.out_subdir

    print(f"Filter: drop if |z_score| > {Z_THRESHOLD} AND all attribution scores <= 0")
    print(f"Output dir: {out_dir}")
    print(f"Hold-out valid: {args.valid_size} records from cleaned train (seed={args.valid_seed})\n")

    for name in args.files:
        src = src_dir / name
        if not src.exists():
            print(f"  SKIP (missing): {src}")
            continue

        kept_lines, dropped = read_clean(src)
        kept = len(kept_lines)
        total = kept + dropped
        pct = 100 * dropped / total if total else 0
        print(f"  {name}: kept={kept}  dropped={dropped}  ({pct:.1f}%)")

        if name == 'train.jsonl':
            rng = random.Random(args.valid_seed)
            valid_idx = set(rng.sample(range(kept), args.valid_size))
            train_lines = [l for i, l in enumerate(kept_lines) if i not in valid_idx]
            valid_lines = [kept_lines[i] for i in sorted(valid_idx)]
            train_path = out_dir / 'train.jsonl'
            valid_path = out_dir / f'valid_subset{args.valid_size}.jsonl'
            write_lines(train_path, train_lines)
            write_lines(valid_path, valid_lines)
            print(f"    -> {train_path}  ({len(train_lines)} records)")
            print(f"    -> {valid_path}  ({len(valid_lines)} records)")
        else:
            dst = out_dir / name
            write_lines(dst, kept_lines)
            print(f"    -> {dst}")


if __name__ == '__main__':
    main()
