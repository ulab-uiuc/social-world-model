import argparse
import json

import jsonlines


def main():
    parser = argparse.ArgumentParser(description='Categorize data based on tags')
    parser.add_argument(
        '--input_file_name',
        type=str,
        default='../data/consensus_data.jsonl',
        help='Path to the input JSONL file containing the data',
    )
    parser.add_argument(
        '--output_file_folder',
        type=str,
        default='../data',
        help='Directory to save the categorized output files',
    )
    args = parser.parse_args()

    categories = ['Politics', 'Sports', 'Crypto', 'Election']
    categorized_data = {category.lower(): [] for category in categories}
    categorized_data['other'] = []

    with open(args.input_file_name, 'r') as input_file:
        for line in input_file:
            entry = json.loads(line)
            tags = entry.get('tags', [])
            categorized = False

            for tag in tags:
                for category in categories:
                    if category.lower() in tag.lower():
                        categorized_data[category.lower()].append(entry)
                        categorized = True
                        break
                if categorized:
                    break

            if not categorized:
                categorized_data['other'].append(entry)

    for category, entries in categorized_data.items():
        output_file = f'{args.output_file_folder}/consensus_data_{category}.jsonl'
        with jsonlines.open(output_file, 'w') as writer:
            writer.write_all(entries)
        print(f'Saved {len(entries)} entries to {output_file}')


if __name__ == '__main__':
    main()
