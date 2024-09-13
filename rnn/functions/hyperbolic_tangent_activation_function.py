from typing import Any

import numpy as np

from rnn.functions.activation_function import ActivationFunction


class HyperbolicTangentActivationFunction(ActivationFunction):

    def __init__(self) -> None:
        super().__init__()

    def value(self, input_data: Any) -> Any:
        return np.tanh(input_data)

    def derived(self, input_data: Any) -> Any:
        return 1 - np.tanh(input_data) ** 2
