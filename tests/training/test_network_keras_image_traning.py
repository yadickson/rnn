from typing import List
from unittest import TestCase

import matplotlib.pyplot as plt
import pytest
from keras.datasets import mnist
from keras.utils import to_categorical

from rnn.data.statistic_initialize_data import StatisticInitializeData
from rnn.functions.hyperbolic_tangent_activation_function import \
    HyperbolicTangentActivationFunction
from rnn.functions.mean_squared_error_loss_function import \
    MeanSquaredErrorLossFunction
from rnn.layers.activation_function_layer import ActivationFunctionLayer
from rnn.layers.fully_connected_layer import FullyConnectedLayer
from rnn.layers.layer import Layer
from rnn.network import Network


@pytest.mark.skipif(reason="never run")
class TestNetworkKerasImageTraining(TestCase):

    layers: List[Layer] = []

    x_test = None
    y_test = None

    network = None
    errors = None

    @classmethod
    def setUpClass(cls):
        learning_rate = 0.1
        # load MNIST from server
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # training data : 60000 samples
        # reshape and normalize input data
        x_train = x_train.reshape(x_train.shape[0], 1, 28 * 28)
        x_train = x_train.astype("float32")
        x_train /= 255
        # encode output which is a number in range [0,9] into a vector of size 10
        # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        y_train = to_categorical(y_train)

        # same for test data : 10000 samples
        x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28)
        x_test = x_test.astype("float32")

        cls.x_test = x_test / 255
        cls.y_test = to_categorical(y_test)

        input_training = x_train
        output_training = y_train

        cls.layers = [
            FullyConnectedLayer(StatisticInitializeData(28 * 28, 50), learning_rate),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
            FullyConnectedLayer(StatisticInitializeData(50, 20), learning_rate),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
            FullyConnectedLayer(StatisticInitializeData(20, 10), learning_rate),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
        ]

        cls.network = Network(layers=cls.layers, loss_function=MeanSquaredErrorLossFunction())
        cls.errors = cls.network.training(input_training, output_training, 50)

    @classmethod
    def tearDownClass(cls):

        epoch = list(range(0, len(cls.errors)))

        plt.cla()
        plt.plot(epoch, cls.errors)
        plt.show()

    def test_should_check_first_image(self):
        result = self.network.process(self.x_test[0])[-1].tolist()
        self.assertEqual([1 if data > 0.99 else 0 for data in result], self.y_test[0].tolist())

    def test_should_check_second_image(self):
        result = self.network.process(self.x_test[1])[-1].tolist()
        self.assertEqual([1 if data > 0.99 else 0 for data in result], self.y_test[1].tolist())

    def test_should_check_third_image(self):
        result = self.network.process(self.x_test[2])[-1].tolist()
        self.assertEqual([1 if data > 0.99 else 0 for data in result], self.y_test[2].tolist())

    def test_layers(self):
        training = [trained.get_trained_values() for trained in self.layers]
        filter(None, training)
        print(training)
