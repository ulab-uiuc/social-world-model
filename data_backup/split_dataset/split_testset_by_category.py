import logging
from collections import defaultdict
from pathlib import Path

import jsonlines

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)

TARGET_CATEGORIES = {
    'Sports': 'swm_bench_Sports_test_new.jsonl',
    'Other': 'swm_bench_Other_test_new.jsonl',
    'Election': 'swm_bench_Election_test_new.jsonl',
    'Crypto': 'swm_bench_Crypto_test_new.jsonl',
    'Politics': 'swm_bench_Politics_test_new.jsonl',
}


def main():
    input_file = 'swm_bench_test_new.jsonl'

    if not Path(input_file).exists():
        logging.error(f'Input file does not exist: {input_file}')
        return

    logging.info(f'Starting to process file: {input_file}')

    data = []
    with jsonlines.open(input_file) as reader:
        for line in reader:
            data.append(line)

    logging.info(f'Loaded {len(data)} data entries')

    categorized_data = defaultdict(list)

    stats = {
        'total_items': len(data),
        'items_with_categories': 0,
        'items_without_categories': 0,
        'category_counts': defaultdict(int),
        'multi_category_items': 0,
    }

    for item in data:
        categories = item.get('categories', [])

        if not categories:
            stats['items_without_categories'] += 1
            categorized_data['Other'].append(item)
            stats['category_counts']['Other'] += 1
            continue

        stats['items_with_categories'] += 1

        if len(categories) > 1:
            stats['multi_category_items'] += 1
            logging.info(
                f'Multi-category item: {item.get("market_id", "unknown")} - {categories}'
            )

        added_to_categories = set()
        for category in categories:
            normalized_category = category.capitalize()

            if normalized_category in TARGET_CATEGORIES:
                categorized_data[normalized_category].append(item)
                stats['category_counts'][normalized_category] += 1
                added_to_categories.add(normalized_category)
            else:
                categorized_data['Other'].append(item)
                stats['category_counts']['Other'] += 1
                added_to_categories.add('Other')

        if not added_to_categories:
            categorized_data['Other'].append(item)
            stats['category_counts']['Other'] += 1

    logging.info('=== Data Categorization Statistics ===')
    logging.info(f'Total items: {stats["total_items"]}')
    logging.info(f'Items with categories: {stats["items_with_categories"]}')
    logging.info(f'Items without categories: {stats["items_without_categories"]}')
    logging.info(f'Multi-category items: {stats["multi_category_items"]}')
    logging.info('Item counts by category:')
    for category, count in stats['category_counts'].items():
        logging.info(f'  {category}: {count}')

    for category, data_list in categorized_data.items():
        if category in TARGET_CATEGORIES:
            output_file = TARGET_CATEGORIES[category]

            logging.info(f'Saving {category} category data to: {output_file}')
            logging.info(f'  Data volume: {len(data_list)} entries')

            with jsonlines.open(output_file, mode='w') as writer:
                for item in data_list:
                    writer.write(item)

            logging.info(f'  {category} category data saved successfully')

    logging.info('Data categorization completed!')
    logging.info('Generated files:')
    for category, filename in TARGET_CATEGORIES.items():
        logging.info(f'  - {filename}')


if __name__ == '__main__':
    main()
