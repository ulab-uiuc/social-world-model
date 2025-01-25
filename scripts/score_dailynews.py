# your_script_openai.py

import json
import os
from datetime import datetime

from swm.data import DailyNewsData, PolyMarketData
from swm.utils.reasoner import PolyMarketDailyNewsReasoner


def main():
    # Initialize OpenAI API key
    openai_api_key = os.environ.get('OPENAI_API_KEY')

    # Sample market data
    markets = [
        PolyMarketData(
            event_id='event1',
            market_id='market1',
            question="Will Company X's stock price exceed $150 today?",
            description="Prediction market for Company X's stock price.",
            start_ts=1672531200.0,  # Example Unix timestamp
            end_ts=1672617600.0,  # Example Unix timestamp
            tags=['finance', 'stocks'],
            categories=['Economy'],
            daily_time_series={
                'outcome1': [
                    {'t': int(datetime(2025, 1, 24).timestamp()), 'p': 145.0},
                    {'t': int(datetime(2025, 1, 25).timestamp()), 'p': 1001.0},
                ]
            },
        ),
        PolyMarketData(
            event_id='event2',
            market_id='market2',
            question='Will the price of Oil drop below $70 per barrel?',
            description='Prediction market for oil prices.',
            start_ts=1672531200.0,
            end_ts=1672617600.0,
            tags=['energy', 'commodities'],
            categories=['Economy'],
            daily_time_series={
                'outcome2': [
                    {'t': int(datetime(2025, 1, 24).timestamp()), 'p': 72.0},
                    {'t': int(datetime(2025, 1, 25).timestamp()), 'p': 500.0},
                ]
            },
        ),
    ]

    # Sample news data
    news = [
        DailyNewsData(
            uuid='news1',
            title='Company X Announces Record Profits',
            description='Company X has reported record profits this quarter, exceeding market expectations.',
            date='2025-01-24',
        ),
        DailyNewsData(
            uuid='news2',
            title='Oil Reserves Increase Unexpectedly',
            description='New reports indicate a significant increase in global oil reserves, impacting prices.',
            date='2025-01-24',
        ),
        DailyNewsData(
            uuid='news3',
            title='Economic Forecast Revised for 2025',
            description='The latest economic forecast predicts a robust growth trajectory for the global market.',
            date='2025-01-24',
        ),
    ]

    # Initialize the reasoner with OpenAI API
    reasoner = PolyMarketDailyNewsReasoner(
        openai_api_key=openai_api_key,
        markets=markets,
        news=news,
        top_k=2,
        news_window_days=1,
    )

    # Specify the date to analyze
    analysis_date = '2025-01-25'

    # Analyze market changes
    try:
        scores = reasoner.analyze(analysis_date)
    except Exception as e:
        print(f'An error occurred during analysis: {e}')
        scores = []

    # Output the results
    if scores:
        print(json.dumps(scores, indent=4))
    else:
        print('No scores returned. Please check the input data and model output.')


if __name__ == '__main__':
    main()
