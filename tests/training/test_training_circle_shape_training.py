from rnn.functions.hyperbolic_tangent_activation_function import (
    HyperbolicTangentActivationFunction,
)
from rnn.functions.mean_squared_error_loss_function import (
    MeanSquaredErrorLossFunction,
)

from rnn.layers.activation_function_layer import ActivationFunctionLayer
from rnn.layers.fully_connected_layer import FullyConnectedLayer
from rnn.network import Network
import numpy as np
from unittest import TestCase
import matplotlib.pyplot as plt
import pytest


class TestNetworkCircleShapeTraining(TestCase):

    def setUp(self):
        self.layers = [
            FullyConnectedLayer(2, 4),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
            FullyConnectedLayer(4, 8),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
            FullyConnectedLayer(8, 1),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
        ]

        self.input_training = np.array(
            [
                [[5, 5]],
                [[6, 6]],
                [[7, 7]],
                [[8, 8]],
                [[9, 9]],
                [[10, 10]],
                [[0, 0]],
                [[1, 1]],
                [[2, 2]],
            ]
        )
        self.output_training = np.array(
            [[[1]], [[1]], [[1]], [[1]], [[1]], [[1]], [[0]], [[0]], [[0]]]
        )

        self.network = Network(
            layers=self.layers, loss_function=MeanSquaredErrorLossFunction()
        )
        self.errors = self.network.training(
            self.input_training, self.output_training, 1000, 0.001
        )

    def test_should_check_process_false_when_input_is_zero_zero(self):
        self.assertLess(self.network.process([[0, 0]])[-1][-1][-1], 0.7)

    def test_should_check_process_true_when_input_is_seven_seven(self):
        self.assertGreaterEqual(self.network.process([[7, 7]])[-1][-1][-1], 0.7)

    def test_should_check_process(self):
        print(self.network.process([[6, 6]]))

    @pytest.mark.skipif(reason="never run")
    def test_show_draw(self):

        epoch = list(range(0, len(self.errors)))

        plt.cla()
        plt.plot(epoch, self.errors)
        plt.show()
