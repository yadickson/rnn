from unittest import TestCase

from keras.datasets import mnist
from keras.utils import to_categorical

from rnn.data.memory_initialize_data import MemoryInitializeData
from rnn.file.json_file import JsonFile
from rnn.functions.hyperbolic_tangent_activation_function import \
    HyperbolicTangentActivationFunction
from rnn.functions.mean_squared_error_loss_function import \
    MeanSquaredErrorLossFunction
from rnn.layers.activation_function_layer import ActivationFunctionLayer
from rnn.layers.fully_connected_layer import FullyConnectedLayer
from rnn.network import Network


class TestNetworkKerasImageTrained(TestCase):

    x_test = None
    y_test = None
    network = None

    @classmethod
    def setUpClass(cls):
        (_, _), (x_test, y_test) = mnist.load_data()

        x_test = x_test.reshape(x_test.shape[0], 1, 28 * 28)
        x_test = x_test.astype("float32")

        cls.x_test = x_test / 255
        cls.y_test = to_categorical(y_test)

        trained_list = JsonFile.read("test_network_keras_image_trained.json").trained

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

    def test_should_check_first_image(self):
        result = self.network.process(self.x_test[0])[-1].tolist()
        self.assertEqual([1 if data > 0.9 else 0 for data in result], self.y_test[0].tolist())

    def test_should_check_third_image(self):
        result = self.network.process(self.x_test[2])[-1].tolist()
        self.assertEqual([1 if data > 0.9 else 0 for data in result], self.y_test[2].tolist())
