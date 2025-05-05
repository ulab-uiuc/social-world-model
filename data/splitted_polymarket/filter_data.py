import jsonlines
from tqdm import tqdm

def clean_line(line: str) -> str:
    # remove any Unicode Line/Paragraph separators
    return line.replace('\u2028', ' ').replace('\u2029', ' ')


input_file_names = [
    'polymarket_data_processed_Crypto_test.jsonl',
    'polymarket_data_processed_Sports_test.jsonl',
    'polymarket_data_processed_Election_test.jsonl',
    'polymarket_data_processed_Other_test.jsonl',
    'polymarket_data_processed_Politics_test.jsonl',
    'polymarket_data_processed_test.jsonl',
    'polymarket_data_processed_Crypto_dev.jsonl',
    'polymarket_data_processed_Sports_dev.jsonl',
    'polymarket_data_processed_Election_dev.jsonl',
    'polymarket_data_processed_Other_dev.jsonl',
    'polymarket_data_processed_Politics_dev.jsonl',
    'polymarket_data_processed_dev.jsonl',
    'polymarket_data_processed_Crypto_train.jsonl',
    'polymarket_data_processed_Sports_train.jsonl',
    'polymarket_data_processed_Election_train.jsonl',
    'polymarket_data_processed_Other_train.jsonl',
    'polymarket_data_processed_Politics_train.jsonl',
    'polymarket_data_processed_train.jsonl',
]
output_file_names = [
    'swm_bench_crypto_test.jsonl',
    'swm_bench_sports_test.jsonl',
    'swm_bench_election_test.jsonl',
    'swm_bench_other_test.jsonl',
    'swm_bench_politics_test.jsonl',
    'swm_bench_test.jsonl',
    'swm_bench_crypto_dev.jsonl',
    'swm_bench_sports_dev.jsonl',
    'swm_bench_election_dev.jsonl',
    'swm_bench_other_dev.jsonl',
    'swm_bench_politics_dev.jsonl',
    'swm_bench_dev.jsonl',
    'swm_bench_crypto_train.jsonl',
    'swm_bench_sports_train.jsonl',
    'swm_bench_election_train.jsonl',
    'swm_bench_other_train.jsonl',
    'swm_bench_politics_train.jsonl',
    'swm_bench_train.jsonl',
]

for in_fn, out_fn in tqdm(zip(input_file_names, output_file_names), total=len(input_file_names)):
    # read & clean raw lines, then parse
    with open(in_fn, 'r', encoding='utf-8') as f_in:
        cleaned_lines = (clean_line(l) for l in f_in)
        reader = jsonlines.Reader(cleaned_lines)
        dataset = [obj for obj in reader]

    # drop unwanted fields
    for obj in dataset:
        obj.pop('breakpoint_ts_pairs', None)
        obj.pop('window_series',       None)

    # write out
    with jsonlines.open(out_fn, 'w') as writer:
        writer.write_all(dataset)