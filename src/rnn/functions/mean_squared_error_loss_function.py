from rnn.functions.loss_function import LossFunction
import numpy as np


class MeanSquaredErrorLossFunction(LossFunction):
    def value(self, real_value, output_desired):
        return np.mean(np.power(real_value - output_desired, 2))

    def derived(self, real_value, output_desired):
        return 2 * (output_desired - real_value) / real_value.size
