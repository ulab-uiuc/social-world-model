from typing import List

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_rmse(predictions: List[float], labels: List[float]) -> float:
    return np.sqrt(mean_squared_error(labels, predictions))


def calculate_mae(predictions: List[float], labels: List[float]) -> float:
    return mean_absolute_error(labels, predictions)
