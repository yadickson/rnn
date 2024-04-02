from rnn.functions.activation_function import ActivationFunction
from rnn.layers.layer import Layer


class ActivationFunctionLayer(Layer):

    def __init__(self, function: ActivationFunction):
        super().__init__()
        self.function = function

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.function.value(self.input)
        return self.output

    def backward_propagation(self, output_error):
        return self.function.derived(self.input) * output_error

    def get_trained_values(self):
        return None
