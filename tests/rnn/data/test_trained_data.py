from unittest import TestCase

import numpy as np
from faker import Faker

from rnn.data.trained_data import TrainedData


class TestTrainedData(TestCase):

    def setUp(self):
        self.faker = Faker()
        self.weights = np.random.rand(2, 3)
        self.bias = np.random.rand(3, 2)
        self.generator = TrainedData(self.weights, self.bias)

    def test_should_check_weights_are_assigned(self):
        self.assertEqual(self.weights.tolist(), self.generator.weights.tolist())

    def test_should_check_bias_is_assigned(self):
        self.assertEqual(self.bias.tolist(), self.generator.bias.tolist())

    def test_should_return_json_string(self):
        result = self.generator.get_json_values()
        expected = '"weights": {}, "bias": {}'.format(self.weights.tolist(), self.bias.tolist())
        self.assertEqual("{" + expected + "}", result)
