import os
from unittest import TestCase

from rnn.data.memory_initialize_data import MemoryInitializeData
from rnn.file.json_file import JsonFile
from rnn.functions.hyperbolic_tangent_activation_function import \
    HyperbolicTangentActivationFunction
from rnn.functions.mean_squared_error_loss_function import \
    MeanSquaredErrorLossFunction
from rnn.functions.relu_activation_function import ReluActivationFunction
from rnn.functions.sigmoid_activation_function import SigmoidActivationFunction
from rnn.layers.activation_function_layer import ActivationFunctionLayer
from rnn.layers.fully_connected_layer import FullyConnectedLayer
from rnn.network import Network


class TestNetworkCircleShapeTrained(TestCase):

    network = None

    @classmethod
    def setUpClass(cls):
        current_directory = os.path.dirname(os.path.realpath(__file__))
        current_file_trained = os.path.join(current_directory, "test_network_circle_shape_trained.json")

        trained_list = JsonFile.read(current_file_trained).trained

        initializer = MemoryInitializeData(trained_list)

        layers = [
            FullyConnectedLayer(initializer),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
            FullyConnectedLayer(initializer),
            ActivationFunctionLayer(ReluActivationFunction()),
            FullyConnectedLayer(initializer),
            ActivationFunctionLayer(SigmoidActivationFunction()),
        ]

        cls.network = Network(layers=layers, loss_function=MeanSquaredErrorLossFunction())

    def test_should_check_process_true_when_input_is_zero_zero(self):
        self.assertGreaterEqual(self.network.process([[0, 0]])[-1], [[0.9]])

    def test_should_check_process_true_when_input_is_one_one(self):
        self.assertGreaterEqual(self.network.process([[1, 1]])[-1], [[0.9]])

    def test_should_check_process_true_when_input_is_minus_one_one(self):
        self.assertGreaterEqual(self.network.process([[-1, 1]])[-1], [[0.9]])

    def test_should_check_process_true_when_input_is_one_minus_one(self):
        self.assertGreaterEqual(self.network.process([[1, -1]])[-1], [[0.9]])

    def test_should_check_process_false_when_input_is_two_two(self):
        self.assertLess(self.network.process([[2, 2]])[-1], [[0.1]])

    def test_should_check_process_false_when_input_is_two_minus_two(self):
        self.assertLess(self.network.process([[2, -2]])[-1], [[0.1]])

    def test_should_check_process_false_when_input_is_four_four(self):
        self.assertLess(self.network.process([[4, 4]])[-1], [[0.1]])

    def test_should_check_process_false_when_input_is_minus_four_four(self):
        self.assertLess(self.network.process([[-4, 0]])[-1], [[0.1]])
