from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from typing import List

def calculate_rmse(predictions: List[float], labels: List[float]) -> float:
    return np.sqrt(mean_squared_error(labels, predictions))

def calculate_mae(predictions: List[float], labels: List[float]) -> float:
    return mean_absolute_error(labels, predictions)