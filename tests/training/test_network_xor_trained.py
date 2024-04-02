from unittest import TestCase

from rnn.data.memory_initialize_data import MemoryInitializeData
from rnn.data.trained_data import TrainedData
from rnn.functions.hyperbolic_tangent_activation_function import \
    HyperbolicTangentActivationFunction
from rnn.functions.mean_squared_error_loss_function import \
    MeanSquaredErrorLossFunction
from rnn.layers.activation_function_layer import ActivationFunctionLayer
from rnn.layers.fully_connected_layer import FullyConnectedLayer
from rnn.network import Network


class TestNetworkXorTrained(TestCase):

    network = None

    @classmethod
    def setUpClass(cls):

        layer_trained_one = TrainedData(
            [
                [0.5814903207629254, -1.6489887269219625, 1.826684882027024],
                [-0.39140716789806207, -1.541773004411403, 1.93996803542501],
            ],
            [[-0.5141473620586524, 2.4670041845295994, -0.6142652324350433]],
        )

        layer_trained_two = TrainedData(
            [[0.2948348030089838], [2.18801507187804], [2.2444522305402215]],
            [[-0.788982573645013]],
        )

        trained_list = [
            layer_trained_one,
            layer_trained_two,
        ]

        initializer = MemoryInitializeData(trained_list)

        layers = [
            FullyConnectedLayer(initializer),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
            FullyConnectedLayer(initializer),
            ActivationFunctionLayer(HyperbolicTangentActivationFunction()),
        ]

        cls.network = Network(layers=layers, loss_function=MeanSquaredErrorLossFunction())

    def test_should_check_process_false_when_input_is_zero_zero(self):
        self.assertLess(self.network.process([[0, 0]])[-1], [[0.1]])

    def test_should_check_process_true_when_input_is_one_zero(self):
        self.assertGreaterEqual(self.network.process([[1, 0]])[-1], [[0.9]])

    def test_should_check_process_true_when_input_is_zero_one(self):
        self.assertGreaterEqual(self.network.process([[0, 1]])[-1], [[0.9]])

    def test_should_check_process_false_when_input_is_one_one(self):
        self.assertLess(self.network.process([[1, 1]])[-1], [[0.1]])
