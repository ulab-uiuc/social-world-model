import jsonlines

from swm.utils.converter import convert_polymarket_event_into_consensus

"""
with jsonlines.open('data_with_offset_0112_with_history.jsonl') as reader:
    dataset_1 = list(reader)

with jsonlines.open('data_with_offset_0112_continue_with_history.jsonl') as reader:
    dataset_2 = list(reader)

merged_dataset = []
unmerged_dataset = dataset_1 + dataset_2
ids = set()
for data in unmerged_dataset:
    if data['id'] not in ids:
        merged_dataset.append(data)
        ids.add(data['id'])

# sort by id
merged_dataset.sort(key=lambda x: x['id'])

with jsonlines.open('data_with_offset_0112_merged_with_history.jsonl', 'w') as writer:
    for data in merged_dataset:
        writer.write(data)
"""

with jsonlines.open('data_with_offset_0112_merged_with_history.jsonl') as reader:
    dataset = list(reader)


valid_market_cnt = 0
valid_and_closed_market_cnt = 0
tot_market_cnt = 0
for data in dataset:
    if 'markets' not in data:
        continue
    markets = data['markets']
    closed_or_not = data['closed']
    for market in markets:
        tot_market_cnt += 1
        history = market['history']
        if len(history) > 0:
            print(len(history))
            valid_market_cnt += 1
            if closed_or_not:
                valid_and_closed_market_cnt += 1

print(tot_market_cnt)
print(valid_market_cnt)
print(valid_and_closed_market_cnt)

consensus = convert_polymarket_event_into_consensus(dataset[0])
