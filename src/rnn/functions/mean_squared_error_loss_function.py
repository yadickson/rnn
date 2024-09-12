from typing import Any

import numpy as np

from rnn.functions.loss_function import LossFunction


class MeanSquaredErrorLossFunction(LossFunction):
    def value(self, real_value: Any, calculated_value: Any) -> Any:
        return np.mean((calculated_value - real_value) ** 2)

    def derived(self, real_value: Any, calculated_value: Any) -> Any:
        return calculated_value - real_value
