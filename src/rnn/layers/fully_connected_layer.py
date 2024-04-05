import numpy as np

from rnn.data.initialize_data import InitializeData
from rnn.layers.layer import Layer


class FullyConnectedLayer(Layer):

    def __init__(self, initializer: InitializeData):
        super().__init__()
        self.trained_values = initializer.get_next_trained_data()
        self.weights = self.trained_values.weights
        self.bias = self.trained_values.bias

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = output_error @ self.weights.T
        weights_error = np.array(self.input).T @ output_error
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error.mean()
        return input_error

    def get_trained_values(self):
        return self.trained_values.get_values()
