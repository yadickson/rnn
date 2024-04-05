from dataclasses import dataclass

import numpy as np


@dataclass
class TrainedData:

    weights: np.ndarray
    bias: np.ndarray

    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = np.array(bias)

    def get_values(self):
        return {
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
        }
