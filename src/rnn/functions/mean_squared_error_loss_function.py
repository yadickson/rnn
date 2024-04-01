import numpy as np

from rnn.functions.loss_function import LossFunction


class MeanSquaredErrorLossFunction(LossFunction):
    def value(self, real_value, calculated_value):
        return np.mean((calculated_value - real_value) ** 2)

    def derived(self, real_value, calculated_value):
        return calculated_value - real_value
