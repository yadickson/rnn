from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TrainedData:

    weights: Any
    bias: Any

    def __init__(self, weights: Any, bias: Any) -> None:
        self.weights = np.array(weights)
        self.bias = np.array(bias)

    def get_values(self) -> Any:
        return {
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
        }
