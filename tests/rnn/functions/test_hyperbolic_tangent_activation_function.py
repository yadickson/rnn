import os
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import pytest
from faker import Faker

from rnn.functions.hyperbolic_tangent_activation_function import \
    HyperbolicTangentActivationFunction


class TestHyperbolicTangentActivationFunction(TestCase):

    def setUp(self):
        self.faker = Faker()
        self.function = HyperbolicTangentActivationFunction()

    def test_should_return_the_same_value_when_input_value_is_greater_than_zero(self):
        input_value = self.faker.random.randint(100, 200)
        self.assertEqual(1.0, self.function.value(input_data=input_value))

    def test_should_return_zero_value_when_input_is_zero(self):
        self.assertEqual(0, self.function.value(input_data=0))

    def test_should_return_minus_one_value_when_input_value_is_less_than_zero(self):
        input_value = self.faker.random.randint(-200, -100)
        self.assertEqual(-1.0, self.function.value(input_data=input_value))

    def test_should_return_zero_derived_value_when_input_value_is_greater_than_zero(self):
        input_value = self.faker.random.randint(100, 200)
        self.assertEqual(0, self.function.derived(input_data=input_value))

    def test_should_return_zero_derived_value_when_input_value_is_zero(self):
        self.assertEqual(1, self.function.derived(input_data=0))

    def test_should_return_zero_derived_value_when_input_value_is_less_than_zero(self):
        input_value = self.faker.random.randint(-200, -100)
        self.assertEqual(0, self.function.derived(input_data=input_value))

    def test_should_return_values_from_array(self):
        input_value = np.array([[1.8, -1.1], [1.0, 7.3], [0.1, -1.0]])
        expected_value = [[0.94681, -0.8005], [0.76159, 1.0], [0.09967, -0.76159]]

        result = np.round(self.function.value(input_data=input_value), 5)

        self.assertEqual(expected_value, result.tolist())

    def test_should_return_values_derived_from_array(self):
        input_value = np.array([[1.8, -1.1], [1.0, 7.3], [0.1, -1.0]])
        expected_value = [[0.10356, 0.3592], [0.41997, 0.0], [0.99007, 0.41997]]

        result = np.round(self.function.derived(input_data=input_value), 5)

        self.assertEqual(expected_value, result.tolist())

    @pytest.mark.skipif(os.environ.get("TRAINING_TEST") is None, reason="run only in training mode")
    def test_show_draw(self):
        data_range = np.linspace(-10, 10).reshape([50, 1])
        data = self.function.value(input_data=data_range)
        data_derived = self.function.derived(input_data=data_range)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        axes[0].plot(data_range, data)
        axes[1].plot(data_range, data_derived)
        fig.tight_layout()
        plt.show()
