from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from faker import Faker

from rnn.functions.activation_function import ActivationFunction
from rnn.layers.activation_function_layer import ActivationFunctionLayer


class TestActivationFunctionLayer(TestCase):

    def setUp(self) -> None:
        self.faker = Faker()
        self.activation_function_stub = MagicMock(ActivationFunction)

        self.activation = np.random.rand(100, 200)
        self.derived = np.random.rand(100, 400)

        self.activation_function_stub.value.return_value = self.activation
        self.activation_function_stub.derived.return_value = self.derived

        self.layer = ActivationFunctionLayer(self.activation_function_stub)

    def test_should_check_input_is_assigned_with_none(self) -> None:
        self.assertEqual(None, self.layer.input)

    def test_should_check_output_is_assigned_with_none(self) -> None:
        self.assertEqual(None, self.layer.output)

    def test_should_check_creation_layer_activation_function_assigned(self) -> None:
        self.assertEqual(self.activation_function_stub, self.layer.function)

    def test_should_check_forward_propagation_input_assigned(self) -> None:
        input_data = np.random.rand(100, 200)

        self.layer.forward_propagation(input_data)

        self.assertEqual(self.layer.input.tolist(), input_data.tolist())

    def test_should_check_forward_propagation_output_assigned(self) -> None:
        input_data = np.random.rand(100, 200)

        self.layer.forward_propagation(input_data)

        self.assertEqual(self.layer.output.tolist(), self.activation.tolist())

    def test_should_check_forward_propagation_result_value(self) -> None:
        input_data = np.random.rand(100, 200)

        result = self.layer.forward_propagation(input_data)

        self.assertEqual(result.tolist(), self.activation.tolist())

    def test_should_check_forward_propagation_parameters_for_activation_function_method(self) -> None:
        input_data = np.random.rand(100, 200)

        self.layer.forward_propagation(input_data)

        self.assertEqual(1, self.activation_function_stub.value.call_count)

        calls = self.activation_function_stub.method_calls

        self.assertTrue(calls[0] == ("value", {"input_data": input_data}))

    def test_should_check_backward_propagation_result_value(self) -> None:
        output_error = np.random.rand(100, 400)
        input_data = np.random.rand(100, 200)
        learning_rate = np.random.rand(100, 100)

        self.layer.input = input_data
        result = self.layer.backward_propagation(output_error, learning_rate)

        self.assertEqual(result.tolist(), (self.derived.tolist() * output_error).tolist())

    def test_should_check_backward_propagation_parameters_for_activation_function_method(self) -> None:
        input_value = np.random.rand(100, 400)
        output_error = np.random.rand(100, 400)
        learning_rate = np.random.rand(100, 300)

        self.layer.input = input_value
        self.layer.backward_propagation(output_error, learning_rate)

        self.assertEqual(1, self.activation_function_stub.derived.call_count)

        calls = self.activation_function_stub.method_calls

        self.assertTrue(calls[0] == ("derived", {"input_data": input_value}))

    def test_should_check_input_data_is_not_assigned_when_call_backward_propagation(self) -> None:
        output_error = np.random.rand(100, 400)
        learning_rate = np.random.rand(100, 100)

        self.layer.backward_propagation(output_error, learning_rate)

        self.assertEqual(None, self.layer.input)

    def test_should_check_output_data_is_not_assigned_when_call_backward_propagation(self) -> None:
        output_error = np.random.rand(100, 400)
        learning_rate = np.random.rand(100, 100)

        self.layer.backward_propagation(output_error, learning_rate)

        self.assertEqual(None, self.layer.output)

    def test_should_check_trained_values_none(self) -> None:
        result = self.layer.get_trained_values()

        self.assertEqual(None, result)
