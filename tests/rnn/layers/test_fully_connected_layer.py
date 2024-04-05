from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from faker import Faker

from rnn.data.initialize_data import InitializeData
from rnn.data.trained_data import TrainedData
from rnn.layers.fully_connected_layer import FullyConnectedLayer


class TestFullyConnectedLayer(TestCase):

    def setUp(self):

        self.faker = Faker()

        self.initializer_stub = MagicMock(InitializeData)
        self.trained_data_stub = MagicMock(TrainedData)

        self.weights = np.random.rand(20, 30)
        self.bias = np.random.rand(10, 20)

        self.trained_data_stub.weights = self.weights
        self.trained_data_stub.bias = self.bias

        self.initializer_stub.get_next_trained_data.return_value = self.trained_data_stub

        self.layer = FullyConnectedLayer(self.initializer_stub)

    def test_should_check_input_is_assigned_with_none(self):
        self.assertEqual(None, self.layer.input)

    def test_should_check_output_is_assigned_with_none(self):
        self.assertEqual(None, self.layer.output)

    def test_should_check_bias_array_is_assigned(self):
        self.assertEqual(self.bias.tolist(), self.layer.bias.tolist())

    def test_should_check_weights_array_is_assigned(self):
        self.assertEqual(self.weights.tolist(), self.layer.weights.tolist())

    def test_should_check_get_next_trained_data_was_called_one_time(self):
        self.assertEqual(1, self.initializer_stub.get_next_trained_data.call_count)

    def test_should_check_forward_propagation_parameters_for_activation_function_method(self):
        self.layer.weights = [[1, 2, 3, 4], [5, 6, 7, 8]]
        self.layer.bias = [10, 20, 30, 40]

        input_data = [[100, 200]]

        result = self.layer.forward_propagation(input_data)

        self.assertEqual([[1110, 1420, 1730, 2040]], result.tolist())

    def test_should_check_trained_values_none(self):
        object_reference = self.faker.random.random()

        self.trained_data_stub.get_values.return_value = object_reference

        result = self.layer.get_trained_values()

        self.assertEqual(object_reference, result)
