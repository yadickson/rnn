from unittest import TestCase

import matplotlib.pyplot as plt
import pytest

from rnn.data.statistic_initialize_data import StatisticInitializeData
from rnn.functions.hyperbolic_tangent_activation_function import \
    HyperbolicTangentActivationFunction
from rnn.functions.mean_squared_error_loss_function import \
    MeanSquaredErrorLossFunction
from rnn.layers.activation_function_layer import ActivationFunctionLayer
from rnn.layers.fully_connected_layer import FullyConnectedLayer
from rnn.network import Network


@pytest.mark.skipif(reason="never run")
class TestNetworkXorTraining(TestCase):

    network = None
    errors = None

    @classmethod
    def setUpClass(cls):
        learning_rate = 0.1

        input_training = [[0, 0], [0, 1], [1, 0], [1, 1]]
        output_training = [[0], [1], [1], [0]]

        layers = [
            FullyConnectedLayer(2, 3, StatisticInitializeData(), learning_rate),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
            FullyConnectedLayer(3, 1, StatisticInitializeData(), learning_rate),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
        ]

        cls.network = Network(layers=layers, loss_function=MeanSquaredErrorLossFunction())
        cls.errors = cls.network.training(input_training, output_training, 5000)

    @classmethod
    def tearDownClass(cls):

        epoch = list(range(0, len(cls.errors)))

        plt.cla()
        plt.plot(epoch, cls.errors)
        plt.show()

    def test_should_check_process_false_when_input_is_zero_zero(self):
        self.assertLess(self.network.process([0, 0])[-1], [0.2])

    def test_should_check_process_true_when_input_is_one_zero(self):
        self.assertGreaterEqual(self.network.process([1, 0])[-1], [0.9])

    def test_should_check_process_true_when_input_is_zero_one(self):
        self.assertGreaterEqual(self.network.process([0, 1])[-1], [0.9])

    def test_should_check_process_false_when_input_is_one_one(self):
        self.assertLess(self.network.process([1, 1])[-1], [0.2])
