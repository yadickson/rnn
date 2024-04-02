from unittest import TestCase

from rnn.data.memory_initialize_data import MemoryInitializeData
from rnn.data.trained_data import TrainedData
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

        layer_trained_one = TrainedData(
            [
                [-2.9579840761763525, -0.8917742553232265, 0.5284617540951397, -1.9524002312747528],
                [0.4152241505310949, -1.1572310338146325, -1.830691898933078, 0.2562144841290061],
            ],
            [[2.064820353649417, -1.8941540143598996, 2.654009423195636, 3.542149808502324]],
        )

        layer_trained_two = TrainedData(
            [
                [
                    2.3359011643484355,
                    -0.9739452806639286,
                    1.7671598067612035,
                    0.990508507628303,
                    -0.3467503005641459,
                    -1.751513835703859,
                    -0.07217164679115821,
                    0.43219961030878445,
                ],
                [
                    -4.519592715600665,
                    0.18048259802435995,
                    -2.4779354691484636,
                    0.393383344691618,
                    1.1086704850492899,
                    -0.3793046444355408,
                    0.1121515221530146,
                    3.3376652476619637,
                ],
                [
                    3.380165746457066,
                    0.27345739506159306,
                    1.9902754140416927,
                    -2.4655166057403317,
                    -0.04598519247055049,
                    -1.1148601895558536,
                    -0.5665866498221849,
                    -3.5956401282636863,
                ],
                [
                    2.138341001547699,
                    0.7059973957442667,
                    0.7874598036480487,
                    0.45876330637361845,
                    -0.34602413479026495,
                    -1.7561104837540344,
                    -0.4274742470492558,
                    0.7080921184335591,
                ],
            ],
            [
                [
                    -3.688880594511402,
                    0.5042185262981358,
                    -1.547480860935703,
                    1.4211289244232965,
                    -0.6309846522123393,
                    0.4817652677553636,
                    -0.067256532971749,
                    0.6378384891388671,
                ]
            ],
        )

        layer_trained_three = TrainedData(
            [
                [5.218220143356209],
                [-0.868349464474984],
                [2.7515775230512904],
                [-2.39278371576708],
                [-0.8343748714374882],
                [-2.1498318061031463],
                [0.25799975842114986],
                [-3.9660147780477573],
            ],
            [[-2.940789701627057]],
        )

        trained_list = [
            layer_trained_one,
            layer_trained_two,
            layer_trained_three,
        ]

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

    def test_should_check_process_false_when_input_is_four_four(self):
        self.assertLess(self.network.process([[4, 4]])[-1], [[0.1]])
