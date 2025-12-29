import argparse
import logging
import time
from typing import Tuple

from llm_generator import LLMGenerator
from pymongo import MongoClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('reasoning_generator.log'), logging.StreamHandler()],
)
logger = logging.getLogger('reasoning_generator')

client = MongoClient('mongodb://localhost:27017/')
db = client['electionDB']
cards_collection = db['cards']
option_reasons_collection = db['option_reasons']


def build_question_prompt(question: str, option: str) -> str:
    return f"""Based on current information and trends, consider this prediction market question:

Question: {question}
Option: {option}

Please think step by step about why this option might be correct or incorrect.
After your reasoning, provide your final judgment on whether this is likely to be the correct outcome.

Please structure your answer as:
Reasoning: [your step-by-step analysis]
Answer: [your final conclusion about whether this option is likely correct]"""


def build_summary_prompt(question: str, option: str, reasoning: str) -> str:
    return f"""You previously analyzed this prediction market question:

Question: {question}
Option: {option}

Your detailed reasoning was:
{reasoning}

Now, please provide a clear paragraph that explain why this option might be true or false.
Each statement should be:
- Around 1-2 sentences
- Specific and informative
- Presented without any formatting (no markdown, no bullet points)
- Free of phrases like "Supporting Points:" or any other headers

Simply list the key factors that would influence the outcome for this option.
Start each reason on a new line."""


def extract_reasoning_and_answer(output: str) -> Tuple[str, str]:
    if 'Reasoning:' in output and 'Answer:' in output:
        parts = output.split('Answer:')
        reasoning = parts[0].replace('Reasoning:', '').strip()
        answer = parts[1].strip()
        return reasoning, answer

    for reasoning_marker in ['Reasoning:', 'Analysis:', "Let's think:"]:
        for answer_marker in ['Answer:', 'Conclusion:', 'Final answer:', 'Judgment:']:
            if reasoning_marker in output and answer_marker in output:
                reasoning_part = output.split(answer_marker)[0]
                reasoning = reasoning_part.split(reasoning_marker, 1)[1].strip()
                answer = output.split(answer_marker, 1)[1].strip()
                return reasoning, answer

    logger.warning(
        'Could not extract reasoning and answer cleanly, using full output as reasoning'
    )
    return output.strip(), ''


def generate_and_store_reasons_for_option(
    card_id: str, question: str, option: str, model_name: str = 'gpt-4o-mini'
) -> list:
    """Generate reasoning for a single option and return the reasons list"""
    llm = LLMGenerator(model_name=model_name)
    logger.info(f'Generating reasoning for card {card_id}, option {option} using {model_name}')

    # Check if the reasoning already exists
    existing = option_reasons_collection.find_one(
        {'card_id': card_id, 'option': option, 'model': model_name}
    )
    
    if existing and any(
        reason.get('votes', 0) > 0 for reason in existing.get('reasons', [])
    ):
        logger.info(f'Reasoning already exists with votes for {card_id}/{option}/{model_name}')
        return existing.get('reasons', [])

    prompt = build_question_prompt(question, option)
    response = llm.generate(prompt)
    output = response.get('content', '')

    if not output:
        logger.warning(f'Empty response for option: {option}')
        return []

    reasoning, answer = extract_reasoning_and_answer(output)

    summary_prompt = build_summary_prompt(question, option, reasoning)
    summary_response = llm.generate(summary_prompt)
    summary = summary_response.get('content', '')

    if not summary:
        logger.warning(f'Empty summary for option: {option}')
        return []

    # Process the reasoning text
    bullet_points = []
    lines = [line.strip() for line in summary.split('\n')]
    lines = [line for line in lines if line and len(line) > 10]

    for line in lines:
        clean_line = line.replace('*', '').strip()
        if ':' in clean_line[:20] or clean_line.isupper():
            continue
        bullet_points.append(clean_line)

    if len(bullet_points) < 2 and lines:
        full_text = ' '.join(lines)
        sentences = [
            s.strip() + '.' for s in full_text.split('.') if len(s.strip()) > 15
        ]
        bullet_points = []
        for i in range(0, min(6, len(sentences)), 2):
            if i + 1 < len(sentences):
                point = sentences[i] + ' ' + sentences[i + 1]
            else:
                point = sentences[i]
            bullet_points.append(point)

    bullet_points = bullet_points[:3]
    if not bullet_points:
        bullet_points = [
            "This option's outcome depends on several factors and current market conditions."
        ]

    reasons = []
    for i, point in enumerate(bullet_points[:3], 1):
        clean_point = point.replace('*', '').strip()
        prefixes_to_remove = [
            'Supporting Reasons:',
            'Opposing Reasons:',
            'Supporting Points:',
            'Opposing Points:',
        ]
        for prefix in prefixes_to_remove:
            if clean_point.startswith(prefix):
                clean_point = clean_point[len(prefix) :].strip()

        reasons.append({'reason_id': str(i), 'text': clean_point, 'votes': 1})

    # Store to database
    option_reasons_collection.update_one(
        {'card_id': card_id, 'option': option, 'model': model_name},
        {'$set': {'card_id': card_id, 'option': option, 'model': model_name, 'reasons': reasons}},
        upsert=True,
    )

    logger.info(f'Stored {len(reasons)} reasons for {card_id}/{option}/{model_name}')
    return reasons


def generate_and_store_reasons(
    model_name: str, card_limit: int = None, sleep_time: float = 1.0
):
    llm = LLMGenerator(model_name=model_name)
    logger.info(f'Using model: {model_name}')

    query = {}
    cards = list(cards_collection.find(query, {'_id': 0}))
    if card_limit:
        cards = cards[:card_limit]

    logger.info(f'Found {len(cards)} cards to process')

    for card in cards:
        card_id = card.get('card_id')
        question = card.get('question')
        options = card.get('options', [])

        if not card_id or not question or not options:
            logger.warning(f'Skipping incomplete card: {card_id}')
            continue

        logger.info(f'Processing card: {card_id} - {question}')

        for option_data in options:
            option = option_data.get('option')
            if not option:
                continue

            logger.info(f'  Generating reasoning for option: {option}')

            existing = option_reasons_collection.find_one(
                {'card_id': card_id, 'option': option, 'model': model_name}
            )

            if existing and any(
                reason.get('votes', 0) > 0 for reason in existing.get('reasons', [])
            ):
                logger.info(
                    f'  Option already has reasons with votes, skipping: {option}'
                )
                continue

            prompt = build_question_prompt(question, option)
            response = llm.generate(prompt)
            output = response.get('content', '')

            if not output:
                logger.warning(f'  Empty response for option: {option}')
                continue

            reasoning, answer = extract_reasoning_and_answer(output)

            summary_prompt = build_summary_prompt(question, option, reasoning)
            summary_response = llm.generate(summary_prompt)
            summary = summary_response.get('content', '')

            if not summary:
                logger.warning(f'  Empty summary for option: {option}')
                continue

            bullet_points = []
            lines = [line.strip() for line in summary.split('\n')]
            lines = [line for line in lines if line and len(line) > 10]

            for line in lines:
                clean_line = line.replace('*', '').strip()
                if ':' in clean_line[:20] or clean_line.isupper():
                    continue
                bullet_points.append(clean_line)

            if len(bullet_points) < 2 and lines:
                full_text = ' '.join(lines)
                sentences = [
                    s.strip() + '.' for s in full_text.split('.') if len(s.strip()) > 15
                ]
                bullet_points = []
                for i in range(0, min(6, len(sentences)), 2):
                    if i + 1 < len(sentences):
                        point = sentences[i] + ' ' + sentences[i + 1]
                    else:
                        point = sentences[i]
                    bullet_points.append(point)

            bullet_points = bullet_points[:3]
            if not bullet_points:
                bullet_points = [
                    "This option's outcome depends on several factors and current market conditions."
                ]

            reasons = []
            for i, point in enumerate(bullet_points[:3], 1):
                clean_point = point.replace('*', '').strip()
                prefixes_to_remove = [
                    'Supporting Reasons:',
                    'Opposing Reasons:',
                    'Supporting Points:',
                    'Opposing Points:',
                ]
                for prefix in prefixes_to_remove:
                    if clean_point.startswith(prefix):
                        clean_point = clean_point[len(prefix) :].strip()

                reasons.append({'reason_id': str(i), 'text': clean_point, 'votes': 1})

            option_reasons_collection.update_one(
                {'card_id': card_id, 'option': option, 'model': model_name},
                {'$set': {'card_id': card_id, 'option': option, 'model': model_name, 'reasons': reasons}},
                upsert=True,
            )

            logger.info(f'  Stored {len(reasons)} reasons for option: {option}')
            time.sleep(sleep_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate reasons for prediction market options'
    )
    parser.add_argument('--model', type=str, default='gpt-4o', help='LLM model to use')
    parser.add_argument(
        '--limit', type=int, default=None, help='Limit number of cards to process'
    )
    parser.add_argument(
        '--sleep', type=float, default=1.0, help='Sleep time between API calls'
    )

    args = parser.parse_args()

    generate_and_store_reasons(
        model_name=args.model, card_limit=args.limit, sleep_time=args.sleep
    )
