import numpy as np

from rnn.functions.activation_function import ActivationFunction


class SigmoidActivationFunction(ActivationFunction):

    def __init__(self):
        super().__init__()

    def value(self, input_data):
        return 1 / (1 + np.exp(-input_data))

    def derived(self, input_data):
        return self.value(input_data) * (1 - self.value(input_data))
