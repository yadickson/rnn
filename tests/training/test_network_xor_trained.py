import os
from unittest import TestCase

from rnn.data.memory_initialize_data import MemoryInitializeData
from rnn.file.json_file import JsonFile
from rnn.functions.hyperbolic_tangent_activation_function import HyperbolicTangentActivationFunction
from rnn.functions.mean_squared_error_loss_function import MeanSquaredErrorLossFunction
from rnn.layers.activation_function_layer import ActivationFunctionLayer
from rnn.layers.fully_connected_layer import FullyConnectedLayer
from rnn.network import Network


class TestNetworkXorTrained(TestCase):

    network: Network

    @classmethod
    def setUpClass(cls) -> None:
        current_directory = os.path.dirname(os.path.realpath(__file__))
        current_file_trained = os.path.join(current_directory, "test_network_xor_trained.json")

        trained_list = JsonFile.read(current_file_trained).trained

        initializer = MemoryInitializeData(trained_list)

        layers = [
            FullyConnectedLayer(initializer),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
            FullyConnectedLayer(initializer),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
        ]

        cls.network = Network(layers=layers, loss_function=MeanSquaredErrorLossFunction())

    def test_should_check_process_false_when_input_is_zero_zero(self) -> None:
        self.assertLess(self.network.process([[0, 0]])[-1], [[0.1]])

    def test_should_check_process_true_when_input_is_one_zero(self) -> None:
        self.assertGreaterEqual(self.network.process([[1, 0]])[-1], [[0.8]])

    def test_should_check_process_true_when_input_is_zero_one(self) -> None:
        self.assertGreaterEqual(self.network.process([[0, 1]])[-1], [[0.8]])

    def test_should_check_process_false_when_input_is_one_one(self) -> None:
        self.assertLess(self.network.process([[1, 1]])[-1], [[0.1]])
