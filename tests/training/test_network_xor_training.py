import os
from typing import List
from unittest import TestCase

import matplotlib.pyplot as plt
import pytest

from rnn.data.statistic_initialize_data import StatisticInitializeData
from rnn.data.training_data import TrainingData
from rnn.file.json_file import JsonFile
from rnn.functions.hyperbolic_tangent_activation_function import \
    HyperbolicTangentActivationFunction
from rnn.functions.mean_squared_error_loss_function import \
    MeanSquaredErrorLossFunction
from rnn.layers.activation_function_layer import ActivationFunctionLayer
from rnn.layers.fully_connected_layer import FullyConnectedLayer
from rnn.layers.layer import Layer
from rnn.network import Network


@pytest.mark.skipif(os.environ.get("TRAINING_TEST") is None, reason="run only in training mode")
class TestNetworkXorTraining(TestCase):

    layers: List[Layer] = []
    network = None
    errors = None

    @classmethod
    def setUpClass(cls):
        input_training = [[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]
        output_training = [[[0]], [[1]], [[1]], [[0]]]

        cls.layers = [
            FullyConnectedLayer(StatisticInitializeData(2, 30)),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
            FullyConnectedLayer(StatisticInitializeData(30, 1)),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
        ]

        cls.network = Network(layers=cls.layers, loss_function=MeanSquaredErrorLossFunction())
        cls.errors = cls.network.training(
            input_training, output_training, [TrainingData(2000, 0.01), TrainingData(1000, 0.001)]
        )

    @classmethod
    def tearDownClass(cls):

        epoch = list(range(0, len(cls.errors)))

        plt.cla()
        plt.plot(epoch, cls.errors)
        plt.show()

    def test_create_training_file(self):
        training = [trained.get_trained_values() for trained in self.layers if trained.get_trained_values() is not None]

        self.assertEqual(2, len(training))

        current_directory = os.path.dirname(os.path.realpath(__file__))
        current_file_trained = os.path.join(current_directory, "test_network_xor_trained.json")

        JsonFile.write(current_file_trained, {"trained": training, "process": self.errors})
