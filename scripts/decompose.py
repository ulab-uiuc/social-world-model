import argparse
import json
import os
import sys
from typing import Dict, List

from tqdm import tqdm

from swm.utils.model_prompting import model_prompting

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def parse_assumptions(response: str) -> List[str]:
    """
    Parse the response text to extract a list of assumptions.
    Each assumption should start with a number (e.g., '1. ') in the response.
    """
    assumptions = []
    for line in response.split('\n'):
        line = line.strip()
        if line and (line[0].isdigit() and line[1] == '.'):
            line = line[3:] if line[2] == ' ' else line[2:]
            assumptions.append(line)

    return assumptions

def collect_introductions(input_file: str) -> Dict[str, str]:
    with open(input_file, 'r') as f:
        dataset = json.load(f)

    introductions = {}
    for arxiv_id, paper in dataset.items():
        introductions[arxiv_id] = paper['paper_data']['introduction']

    return introductions


def decompose_assumptions(input_file: str, output_file: str):
    # Load the JSON file with paper introductions
    paper_introductions = collect_introductions(input_file)

    # Dictionary to store assumptions for each paper
    assumptions = {}

    # Iterate through each paper introduction
    for arxiv_id, introduction in tqdm(paper_introductions.items()):
        messages = [
            {"role": "system", "content": "You are an expert AI research assistant."},
            {"role": "user", "content": f"Here's the introduction to a research paper: '{introduction}'. Please identify the assumptions made in this introduction. You need to list assumptions as multiple points. Each point should be a sentence like 'LoRA is an effective technique to conduct parameter-efficient finetuning' and another assumption is like 'High-quality instruction data is more important than the number of data for training a large language model.' You need to list it in the format of 1. <assumption 1>, 2. <assumption 2>, and so on. Please do not say assumptions that you think are not useful or not relevant to the research paper. PLEASE DO NOT SAY WORDS like the assumption is xxx but directly say each assumption as a statement."}
        ]

        # Call the model_prompting function
        response = model_prompting(
            llm_model="gpt-4",
            messages=messages,
            return_num=1,
            max_token_num=512,
            temperature=0.5,
            top_p=0.9
        )

        # Parse the response into a list of assumptions
        assumptions[arxiv_id] = {}
        assumptions[arxiv_id]['introduction'] = introduction
        assumptions[arxiv_id]['assumptions'] = parse_assumptions(response[0]) if response else ["No assumptions identified."]

        # Incrementally save to output file after each paper is processed
        with open(output_file, 'w') as f:
            json.dump(assumptions, f, indent=4)

    print(f"Assumptions analysis saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze assumptions from paper introductions.")
    parser.add_argument("--input_file", type=str, default='../data/iclrbench.json', help="Path to the input JSON file containing paper introductions.")
    parser.add_argument("--output_file", type=str, default='../data/iclrbench_decompose.json', help="Path to save the output JSON file with assumptions.")

    args = parser.parse_args()
    decompose_assumptions(args.input_file, args.output_file)
