import json
import jsonlines
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str,
                    default="data/consensus_data.jsonl")
args = parser.parse_args()

tags = ['Politics', 'Sports', 'Crypto', 'Election']
dict = {}

for tmp in tags:
    dict[tmp.lower()] = []
dict['other'] = []

with open(args.file_name, 'r') as fcc_file:
    for line in fcc_file:
        obj = json.loads(line)
        tag_ls = obj['tags']
        flag = False
        for t in tag_ls:
            for tmp in tags:
                if tmp.lower() in t.lower():
                    dict[tmp.lower()].append(obj)
                    flag = True
                    break
        if flag == False: dict['other'].append(obj)


for k in dict.keys():
    with jsonlines.open(f'data/consensus_data_{k}.jsonl', 'w') as writer:
        writer.write_all(dict[k])
