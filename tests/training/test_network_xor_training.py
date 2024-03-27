from unittest import TestCase

import numpy as np
import pytest

from rnn.functions.hyperbolic_tangent_activation_function import (
    HyperbolicTangentActivationFunction,
)
from rnn.functions.mean_squared_error_loss_function import (
    MeanSquaredErrorLossFunction,
)
from rnn.layers.activation_function_layer import ActivationFunctionLayer
from rnn.layers.fully_connected_layer import FullyConnectedLayer
from rnn.network import Network
import matplotlib.pyplot as plt


class TestNetworkXorTraining(TestCase):

    def setUp(self):
        self.layers = [
            FullyConnectedLayer(2, 3),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
            FullyConnectedLayer(3, 1),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
        ]

        self.input_training = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
        self.output_training = np.array([[[0]], [[1]], [[1]], [[0]]])

        self.network = Network(
            layers=self.layers, loss_function=MeanSquaredErrorLossFunction()
        )
        self.errors = self.network.training(
            self.input_training, self.output_training, 5000, 0.01
        )

    def test_should_check_process_false_when_input_is_zero_zero(self):
        self.assertLess(self.network.process([[0, 0]])[-1][-1][-1], 0.8)

    def test_should_check_process_true_when_input_is_one_zero(self):
        self.assertGreaterEqual(self.network.process([[1, 0]])[-1][-1][-1], 0.8)

    def test_should_check_process_true_when_input_is_zero_one(self):
        self.assertGreaterEqual(self.network.process([[0, 1]])[-1][-1][-1], 0.8)

    def test_should_check_process_false_when_input_is_one_one(self):
        self.assertLess(self.network.process([[1, 1]])[-1][-1][-1], 0.8)

    @pytest.mark.skipif(reason="never run")
    def test_show_draw(self):

        epoch = list(range(0, len(self.errors)))

        plt.cla()
        plt.plot(epoch, self.errors)
        plt.show()
