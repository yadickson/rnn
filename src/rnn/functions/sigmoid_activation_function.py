from rnn.functions.activation_function import ActivationFunction
import numpy as np


class SigmoidActivationFunction(ActivationFunction):

    def __init__(self):
        super().__init__()

    def value(self, input_data):
        return np.round(1 / (1 + np.exp(-input_data)), 5)

    def derived(self, input_data):
        return input_data * (1 - input_data)
