import math
import os
from typing import List
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import stats

from rnn.data.statistic_initialize_data import StatisticInitializeData
from rnn.data.training_data import TrainingData
from rnn.file.json_file import JsonFile
from rnn.functions.hyperbolic_tangent_activation_function import \
    HyperbolicTangentActivationFunction
from rnn.functions.mean_squared_error_loss_function import \
    MeanSquaredErrorLossFunction
from rnn.functions.relu_activation_function import ReluActivationFunction
from rnn.functions.sigmoid_activation_function import SigmoidActivationFunction
from rnn.layers.activation_function_layer import ActivationFunctionLayer
from rnn.layers.fully_connected_layer import FullyConnectedLayer
from rnn.layers.layer import Layer
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


@pytest.mark.skipif(os.environ.get("TRAINING_TEST") is None, reason="run only in training mode")
class TestNetworkCircleShapeTraining(TestCase):

    count = None
    X = None
    Y = None

    layers: List[Layer] = []
    network = None
    errors = None

    @classmethod
    def setUpClass(cls):

        cls.count = 500

        data_not_ok = circle(count=cls.count, Ri=2, Ro=2)
        data_ok = circle(count=cls.count, Ri=2, Ro=0)

        cls.X = np.concatenate([data_not_ok, data_ok])
        cls.Y = [[0]] * cls.count + [[1]] * cls.count

        input_training = [[data] for data in cls.X]
        output_training = [[data] for data in cls.Y]

        cls.layers = [
            FullyConnectedLayer(StatisticInitializeData(2, 40)),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
            FullyConnectedLayer(StatisticInitializeData(40, 80)),
            ActivationFunctionLayer(ReluActivationFunction()),
            FullyConnectedLayer(StatisticInitializeData(80, 1)),
            ActivationFunctionLayer(SigmoidActivationFunction()),
        ]

        cls.network = Network(layers=cls.layers, loss_function=MeanSquaredErrorLossFunction())
        cls.errors = cls.network.training(
            input_training,
            output_training,
            [TrainingData(200, 0.1), TrainingData(100, 0.01), TrainingData(50, 0.001), TrainingData(20, 0.0001)],
        )

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

    def test_create_training_file(self):
        training = [trained.get_trained_values() for trained in self.layers if trained.get_trained_values() is not None]

        self.assertEqual(3, len(training))

        current_directory = os.path.dirname(os.path.realpath(__file__))
        current_file_trained = os.path.join(current_directory, "test_network_circle_shape_trained.json")

        JsonFile.write(current_file_trained, {"trained": training, "process": self.errors})
