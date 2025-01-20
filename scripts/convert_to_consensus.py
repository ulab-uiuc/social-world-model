import jsonlines

from swm.data import Consensus
from swm.utils.converter import (
    find_action_in_states,
    find_state_action_pairs,
    find_state_change_at_timestamp,
)

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


"""
with jsonlines.open('../data/data_with_offset_0112_merged_with_history.jsonl') as reader:
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
            valid_market_cnt += 1
            if closed_or_not:
                valid_and_closed_market_cnt += 1

print(tot_market_cnt)
print(valid_market_cnt)
print(valid_and_closed_market_cnt)
print(len(dataset))

for data in tqdm(dataset):
    consensuses = convert_polymarket_event_into_consensuses(data)
    with jsonlines.open('../data/consensus_data.jsonl', 'a') as writer:
        for consensus in consensuses:
            consensus_json = consensus.model_dump()
            writer.write(consensus_json)
"""

with jsonlines.open('../data/consensus_data.jsonl') as reader:
    dataset = list(reader)

consensuses = []
for data in dataset:
    consensus = Consensus.from_dict(data)
    consensuses.append(consensus)

related_consenses = find_state_change_at_timestamp(
    ts=1730851202, consensuses=consensuses
)


actions = find_action_in_states(consensuses)

state_action_pairs = find_state_action_pairs(actions=actions, consensuses=consensuses)
print(len(state_action_pairs))
