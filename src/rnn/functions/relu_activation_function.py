import numpy as np

from rnn.functions.activation_function import ActivationFunction


class ReluActivationFunction(ActivationFunction):

    def __init__(self):
        super().__init__()

    def value(self, input_data):
        return input_data * (input_data > 0)

    def derived(self, input_data):
        return np.round(1 / (1 + np.exp(-input_data)), 0)
