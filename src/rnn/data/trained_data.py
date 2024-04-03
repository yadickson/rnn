from dataclasses import dataclass

from numpy import ndarray


@dataclass
class TrainedData:

    weights: ndarray
    bias: ndarray

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def get_values(self):
        return {
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
        }
