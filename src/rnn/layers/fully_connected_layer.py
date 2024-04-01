import numpy as np

from rnn.data.initialize_data import InitializeData
from rnn.layers.layer import Layer


class FullyConnectedLayer(Layer):

    def __init__(self, input_size: int, output_size: int, initializer: InitializeData, learning_rate: float):
        super().__init__()
        self.weights = initializer.create(input_size=input_size, output_size=output_size)
        self.bias = initializer.create(input_size=1, output_size=output_size)
        self.learning_rate = learning_rate

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error):
        input_error = output_error @ self.weights.T
        weights_error = np.array(self.input).T @ output_error
        self.weights -= self.learning_rate * weights_error
        self.bias -= self.learning_rate * output_error
        return input_error
