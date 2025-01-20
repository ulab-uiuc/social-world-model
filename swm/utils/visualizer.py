import datetime
from typing import Dict, List, Union

import matplotlib.pyplot as plt


def visualize_price_history(
    history: List[Dict[str, Union[int, float]]],
    title: str = 'Price History',
    save_path: str = None,
):
    timestamps = [item['t'] for item in history]
    prices = [item['p'] for item in history]
    datetimes = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]

    plt.figure(figsize=(10, 6))
    plt.plot(datetimes, prices, marker='o', linestyle='-', color='b')
    plt.xlabel('Date/Time')
    plt.ylabel('Price')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()
