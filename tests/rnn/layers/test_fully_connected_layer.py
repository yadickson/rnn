from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from faker import Faker

from rnn.data.initialize_data import InitializeData
from rnn.layers.fully_connected_layer import FullyConnectedLayer


class TestFullyConnectedLayer(TestCase):

    def setUp(self):

        self.faker = Faker()

        self.input_size = self.faker.random.randint(20, 30)
        self.output_size = self.faker.random.randint(40, 50)
        self.learning_rate = self.faker.random.random()

        self.initializer_stub = MagicMock(InitializeData)

        self.weights = np.random.rand(20, 30)
        self.bias = np.random.rand(10, 20)

        values = [self.weights, self.bias]

        self.initializer_stub.create.side_effect = values

        self.layer = FullyConnectedLayer(
            self.input_size,
            self.output_size,
            self.initializer_stub,
            self.learning_rate,
        )

    def test_should_check_input_is_assigned_with_none(self):
        self.assertEqual(None, self.layer.input)

    def test_should_check_output_is_assigned_with_none(self):
        self.assertEqual(None, self.layer.output)

    def test_should_check_learning_rate_is_assigned(self):
        self.assertEqual(self.learning_rate, self.layer.learning_rate)

    def test_should_check_bias_array_is_assigned(self):
        self.assertEqual(self.bias.tolist(), self.layer.bias.tolist())

    def test_should_check_weights_array_is_assigned(self):
        self.assertEqual(self.weights.tolist(), self.layer.weights.tolist())

    def test_should_check_creation_layer_bias_and_weights_are_assigned(self):
        self.assertEqual(2, self.initializer_stub.create.call_count)

        calls = self.initializer_stub.create.mock_calls

        calls[0].assert_called_with(input_size=self.input_size, output_size=self.output_size)
        calls[1].assert_called_with(input_size=1, output_size=self.output_size)

    def test_should_check_forward_propagation_parameters_for_activation_function_method(self):
        self.layer.weights = [[1, 2, 3, 4], [5, 6, 7, 8]]
        self.layer.bias = [10, 20, 30, 40]

        input_data = [[100, 200]]

        result = self.layer.forward_propagation(input_data)

        self.assertEqual([[1110, 1420, 1730, 2040]], result.tolist())
