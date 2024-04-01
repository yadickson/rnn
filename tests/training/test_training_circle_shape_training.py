import math
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import stats

from rnn.data.statistic_initialize_data import StatisticInitializeData
from rnn.functions.hyperbolic_tangent_activation_function import \
    HyperbolicTangentActivationFunction
from rnn.functions.mean_squared_error_loss_function import \
    MeanSquaredErrorLossFunction
from rnn.functions.relu_activation_function import ReluActivationFunction
from rnn.functions.sigmoid_activation_function import SigmoidActivationFunction
from rnn.layers.activation_function_layer import ActivationFunctionLayer
from rnn.layers.fully_connected_layer import FullyConnectedLayer
from rnn.network import Network


def circle(count=100, Ri=1, Ro=2, min=0, max=1):

    r = Ri * np.sqrt(stats.truncnorm.rvs(min, max, size=count))
    theta = stats.truncnorm.rvs(min, max, size=count) * 2 * math.pi

    x = np.cos(theta) * (r + Ro)
    y = np.sin(theta) * (r + Ro)

    y = y.reshape((count, 1))
    x = x.reshape((count, 1))

    df = np.column_stack([x, y])
    return df


@pytest.mark.skipif(reason="never run")
class TestNetworkCircleShapeTraining(TestCase):

    count = None
    X = None
    Y = None
    network = None
    errors = None

    @classmethod
    def setUpClass(cls):

        learning_rate = 0.1
        cls.count = 100

        data_not_ok = circle(count=cls.count, Ri=2, Ro=2)
        data_ok = circle(count=cls.count, Ri=2, Ro=0)

        cls.X = np.concatenate([data_not_ok, data_ok])
        cls.Y = [0] * cls.count + [1] * cls.count

        input_training = cls.X
        output_training = [[data] for data in cls.Y]

        layers = [
            FullyConnectedLayer(2, 4, StatisticInitializeData(), learning_rate),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
            FullyConnectedLayer(4, 8, StatisticInitializeData(), learning_rate),
            ActivationFunctionLayer(ReluActivationFunction()),
            FullyConnectedLayer(8, 1, StatisticInitializeData(), learning_rate),
            ActivationFunctionLayer(SigmoidActivationFunction()),
        ]

        cls.network = Network(layers=layers, loss_function=MeanSquaredErrorLossFunction())
        cls.errors = cls.network.training(input_training, output_training, 5000)

    @classmethod
    def tearDownClass(cls):

        plt.cla()
        plt.scatter(cls.X[0 : cls.count, 0], cls.X[0 : cls.count, 1], c="b")
        plt.scatter(cls.X[cls.count : cls.count * 2, 0], cls.X[cls.count : cls.count * 2, 1], c="r")
        plt.show()

        epoch = list(range(0, len(cls.errors)))

        plt.cla()
        plt.plot(epoch, cls.errors)
        plt.show()

    def test_should_check_process_true_when_input_is_zero_zero(self):
        self.assertGreaterEqual(self.network.process([0, 0])[-1], [0.8])

    def test_should_check_process_false_when_input_is_seven_seven(self):
        self.assertLess(self.network.process([4, 4])[-1], [0.2])
