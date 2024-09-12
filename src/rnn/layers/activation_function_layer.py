from typing import Any

from rnn.functions.activation_function import ActivationFunction
from rnn.layers.layer import Layer


class ActivationFunctionLayer(Layer):

    def __init__(self, function: ActivationFunction) -> None:
        super().__init__()
        self.function = function

    def forward_propagation(self, input_data: Any) -> Any:
        self.input = input_data
        self.output = self.function.value(input_data=self.input)
        return self.output

    def backward_propagation(self, output_error: Any, learning_rate: Any) -> Any:
        return self.function.derived(input_data=self.input) * output_error

    def get_trained_values(self) -> Any:
        return None
