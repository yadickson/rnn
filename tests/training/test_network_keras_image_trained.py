import os
from typing import Any
from unittest import TestCase

from keras.datasets import mnist
from keras.utils import to_categorical

from rnn.data.memory_initialize_data import MemoryInitializeData
from rnn.file.json_file import JsonFile
from rnn.functions.hyperbolic_tangent_activation_function import HyperbolicTangentActivationFunction
from rnn.functions.mean_squared_error_loss_function import MeanSquaredErrorLossFunction
from rnn.layers.activation_function_layer import ActivationFunctionLayer
from rnn.layers.fully_connected_layer import FullyConnectedLayer
from rnn.network import Network


class TestNetworkKerasImageTrained(TestCase):

    x_test: Any
    y_test: Any
    network: Network

    @classmethod
    def setUpClass(cls) -> None:
        (_, _), (x_test, y_test) = mnist.load_data()

        x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28)
        x_test = x_test.astype("float32")

        cls.x_test = (x_test / 255)[0:1000]
        cls.y_test = to_categorical(y_test)[0:1000]

        current_directory = os.path.dirname(os.path.realpath(__file__))
        current_file_trained = os.path.join(current_directory, "test_network_keras_image_trained.json")

        trained_list = JsonFile.read(current_file_trained).trained

        initializer = MemoryInitializeData(trained_list)

        layers = [
            FullyConnectedLayer(initializer),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
            FullyConnectedLayer(initializer),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
            FullyConnectedLayer(initializer),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
        ]

        cls.network = Network(layers=layers, loss_function=MeanSquaredErrorLossFunction())

    def test_should_check_first_image(self) -> None:
        result = self.network.process(self.x_test[0])
        expected = (result == result.max(axis=1)[:, None]).astype(int)[-1]
        self.assertEqual(expected.tolist(), self.y_test[0].tolist())

    def test_should_check_second_image(self) -> None:
        result = self.network.process(self.x_test[1])
        expected = (result == result.max(axis=1)[:, None]).astype(int)[-1]
        self.assertEqual(expected.tolist(), self.y_test[1].tolist())

    def test_should_check_third_image(self) -> None:
        result = self.network.process(self.x_test[2])
        expected = (result == result.max(axis=1)[:, None]).astype(int)[-1]
        self.assertEqual(expected.tolist(), self.y_test[2].tolist())

    def test_should_check_full_images(self) -> None:
        for index in range(7):
            result = self.network.process(self.x_test[index])
            expected = (result == result.max(axis=1)[:, None]).astype(int)[-1]
            self.assertEqual(expected.tolist(), self.y_test[index].tolist())
