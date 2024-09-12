from typing import Any

import numpy as np

from rnn.functions.activation_function import ActivationFunction


class SigmoidActivationFunction(ActivationFunction):

    def __init__(self) -> None:
        super().__init__()

    def value(self, input_data: Any) -> Any:
        return 1 / (1 + np.exp(-input_data))

    def derived(self, input_data: Any) -> Any:
        return self.value(input_data) * (1 - self.value(input_data))
