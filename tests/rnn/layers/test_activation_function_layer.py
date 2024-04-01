from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from faker import Faker

from rnn.functions.activation_function import ActivationFunction
from rnn.layers.activation_function_layer import ActivationFunctionLayer


class TestActivationFunctionLayer(TestCase):

    def setUp(self):
        self.faker = Faker()
        self.activation_function_stub = MagicMock(ActivationFunction)

        self.activation = np.random.rand(100, 200)
        self.derived = np.random.rand(100, 400)

        self.activation_function_stub.value.return_value = self.activation
        self.activation_function_stub.derived.return_value = self.derived

        self.layer = ActivationFunctionLayer(self.activation_function_stub)

    def test_should_check_input_is_assigned_with_none(self):
        self.assertEqual(None, self.layer.input)

    def test_should_check_output_is_assigned_with_none(self):
        self.assertEqual(None, self.layer.output)

    def test_should_check_creation_layer_activation_function_assigned(self):
        self.assertEqual(self.activation_function_stub, self.layer.function)

    def test_should_check_forward_propagation_input_assigned(self):
        input_data = np.random.rand(100, 200)

        self.layer.forward_propagation(input_data)

        self.assertEqual(self.layer.input.tolist(), input_data.tolist())

    def test_should_check_forward_propagation_output_assigned(self):
        input_data = np.random.rand(100, 200)

        self.layer.forward_propagation(input_data)

        self.assertEqual(self.layer.output.tolist(), self.activation.tolist())

    def test_should_check_forward_propagation_result_value(self):
        input_data = np.random.rand(100, 200)

        result = self.layer.forward_propagation(input_data)

        self.assertEqual(result.tolist(), self.activation.tolist())

    def test_should_check_forward_propagation_parameters_for_activation_function_method(self):
        input_data = np.random.rand(100, 200)

        self.layer.forward_propagation(input_data)

        self.assertEqual(1, self.activation_function_stub.value.call_count)

        calls = self.activation_function_stub.value.mock_calls
        calls[0].assert_called_with(input_data=input_data)

    def test_should_check_backward_propagation_result_value(self):
        output_error = np.random.rand(100, 400)
        input_data = np.random.rand(100, 200)

        self.layer.input = input_data
        result = self.layer.backward_propagation(output_error)

        self.assertEqual(result.tolist(), (self.derived.tolist() * output_error).tolist())

    def test_should_check_backward_propagation_parameters_for_activation_function_method(self):
        output_error = np.random.rand(100, 400)

        self.layer.backward_propagation(output_error)

        self.assertEqual(1, self.activation_function_stub.derived.call_count)

        calls = self.activation_function_stub.derived.mock_calls
        calls[0].assert_called_with(input_data=output_error)

    def test_should_check_input_data_is_not_assigned_when_call_backward_propagation(self):
        output_error = np.random.rand(100, 400)

        self.layer.backward_propagation(output_error)

        self.assertEqual(None, self.layer.input)

    def test_should_check_output_data_is_not_assigned_when_call_backward_propagation(self):
        output_error = np.random.rand(100, 400)

        self.layer.backward_propagation(output_error)

        self.assertEqual(None, self.layer.output)
