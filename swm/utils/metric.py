from typing import List

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_rmse(predictions: List[float], labels: List[float]) -> float:
    return np.sqrt(mean_squared_error(labels, predictions))


def calculate_mae(predictions: List[float], labels: List[float]) -> float:
    return mean_absolute_error(labels, predictions)


def calculate_mse(predictions: List[float], labels: List[float]) -> float:
    return mean_squared_error(labels, predictions)


def calculate_mape(predictions: List[float], labels: List[float]) -> float:
    return (
        np.mean(np.abs((np.array(labels) - np.array(predictions)) / np.array(labels)))
        * 100
    )


def calculate_metric(predictions: List[float], labels: List[float]) -> dict:
    return {
        'rmse': np.sqrt(mean_squared_error(labels, predictions)),
        'mae': calculate_mae(predictions, labels),
        'mse': mean_squared_error(labels, predictions),
        'mape': np.mean(
            np.abs((np.array(labels) - np.array(predictions)) / np.array(labels))
        )
        * 100,
    }
