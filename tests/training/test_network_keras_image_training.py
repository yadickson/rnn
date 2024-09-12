import os
from typing import Any, List
from unittest import TestCase

import matplotlib.pyplot as plt
import pytest
from keras.datasets import mnist
from keras.utils import to_categorical

from rnn.data.statistic_initialize_data import StatisticInitializeData
from rnn.data.training_data import TrainingData
from rnn.file.json_file import JsonFile
from rnn.functions.hyperbolic_tangent_activation_function import HyperbolicTangentActivationFunction
from rnn.functions.mean_squared_error_loss_function import MeanSquaredErrorLossFunction
from rnn.layers.activation_function_layer import ActivationFunctionLayer
from rnn.layers.fully_connected_layer import FullyConnectedLayer
from rnn.layers.layer import Layer
from rnn.network import Network


@pytest.mark.skipif(os.environ.get("TRAINING_TEST") is None, reason="run only in training mode")
class TestNetworkKerasImageTraining(TestCase):

    layers: List[Layer] = []

    network: Network
    errors: Any

    @classmethod
    def setUpClass(cls) -> None:

        # load MNIST from server
        (x_train, y_train), (_, _) = mnist.load_data()

        # training data : 60000 samples
        # reshape and normalize input data
        x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28)
        x_train = x_train.astype("float32")
        x_train /= 255
        # encode output which is a number in range [0,9] into a vector of size 10
        # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        y_train = to_categorical(y_train)

        input_training = x_train[0:1000]
        output_training = y_train[0:1000]

        cls.layers = [
            FullyConnectedLayer(StatisticInitializeData(28 * 28, 50)),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
            FullyConnectedLayer(StatisticInitializeData(50, 20)),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
            FullyConnectedLayer(StatisticInitializeData(20, 10)),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
        ]

        cls.network = Network(layers=cls.layers, loss_function=MeanSquaredErrorLossFunction())
        cls.errors = cls.network.training(
            input_training,
            output_training,
            [
                TrainingData(10, 0.1),
                TrainingData(10, 0.01),
                TrainingData(5, 0.001),
                TrainingData(4, 0.0001),
                TrainingData(3, 0.00001),
            ],
        )

    @classmethod
    def tearDownClass(cls) -> None:

        epoch = list(range(0, len(cls.errors)))

        plt.cla()
        plt.plot(epoch, cls.errors)
        plt.show()

    def test_create_training_file(self) -> None:
        training = [trained.get_trained_values() for trained in self.layers if trained.get_trained_values() is not None]

        self.assertEqual(3, len(training))

        current_directory = os.path.dirname(os.path.realpath(__file__))
        current_file_trained = os.path.join(current_directory, "test_network_keras_image_trained.json")

        JsonFile.write(current_file_trained, {"trained": training, "process": self.errors})
