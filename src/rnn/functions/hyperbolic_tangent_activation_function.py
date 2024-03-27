from rnn.functions.activation_function import ActivationFunction
import numpy as np


class HyperbolicTangentActivationFunction(ActivationFunction):

    def __init__(self):
        super().__init__()

    def value(self, input_data):
        return np.tanh(input_data)

    def derived(self, input_data):
        return 1 - np.tanh(input_data) ** 2
