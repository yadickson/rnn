from rnn.layer import Layer
import numpy as np
from scipy import stats


class FullyConnectedLayer(Layer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.bias = stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size=output_size)
        self.weights = stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size=output_size * input_size)
        self.bias = self.bias.reshape(1, output_size)
        self.weights = self.weights.reshape(input_size, output_size)

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
