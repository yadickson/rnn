from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from rnn.functions.loss_function import LossFunction
from rnn.layers.layer import Layer
from rnn.network import Network


class TestNetwork(TestCase):

    def setUp(self):
        self.first_layer_stub = MagicMock(Layer)
        self.second_layer_stub = MagicMock(Layer)

        self.first_forward = np.random.rand(2, 3)
        self.first_layer_stub.forward_propagation.return_value = self.first_forward

        self.second_forward = np.random.rand(1, 1)
        self.second_layer_stub.forward_propagation.return_value = self.second_forward

        self.layers = [self.first_layer_stub, self.second_layer_stub]

        self.loss_function_stub = MagicMock(LossFunction)

        self.network = Network(self.layers, self.loss_function_stub)

    def test_should_check_layers_are_assigned(self):
        self.assertEqual(self.layers, self.network.layers)

    def test_should_check_loss_function_is_assigned(self):
        self.assertEqual(self.loss_function_stub, self.network.loss_function)

    def test_should_check_first_layer_forward_propagation(self):

        input_data = np.random.rand(20, 30)

        self.network.process(input_data)

        self.assertEqual(1, self.first_layer_stub.forward_propagation.call_count)

        calls = self.first_layer_stub.forward_propagation.mock_calls
        calls[0].assert_called_with(input_data=input_data)

    def test_should_check_second_layer_forward_propagation(self):

        input_data = np.random.rand(20, 30)

        self.network.process(input_data)

        self.assertEqual(1, self.second_layer_stub.forward_propagation.call_count)

        calls = self.second_layer_stub.forward_propagation.mock_calls
        calls[0].assert_called_with(input_data=self.first_forward)

    def test_should_check_process_response_last_element(self):
        input_data = np.random.rand(20, 30)
        self.assertEqual(self.network.process(input_data), self.second_forward.tolist())
