from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import pytest
from faker import Faker

from rnn.functions.sigmoid_activation_function import SigmoidActivationFunction


class TestSigmoidActivationFunction(TestCase):

    def setUp(self):
        self.faker = Faker()
        self.function = SigmoidActivationFunction()

    def test_should_return_one_value_when_input_value_is_greater_than_zero(self):
        input_value = self.faker.random.randint(100, 200)
        self.assertEqual(1.0, self.function.value(input_data=input_value))

    def test_should_return_not_zero_value_when_input_is_one(self):
        self.assertEqual(0.73106, np.round(self.function.value(input_data=1), 5))

    def test_should_return_zero_dot_five_value_when_input_is_zero(self):
        self.assertEqual(0.5, self.function.value(input_data=0))

    def test_should_return_zero_value_when_input_value_is_less_than_zero(self):
        input_value = self.faker.random.randint(-200, -100)
        self.assertEqual(0, np.round(self.function.value(input_data=input_value), 5))

    def test_should_return_zero_derived_value_when_input_value_is_greater_than_zero(self):
        input_value = self.faker.random.randint(100, 200)
        self.assertEqual(0.0, np.round(self.function.derived(input_data=input_value), 5))

    def test_should_return_zero_derived_value_when_input_value_is_zero(self):
        self.assertEqual(0.25, self.function.derived(input_data=0))

    def test_should_return_zero_derived_value_when_input_value_is_less_than_zero(self):
        input_value = self.faker.random.randint(-200, -100)
        self.assertEqual(0, np.round(self.function.derived(input_data=input_value), 5))

    def test_should_return_values_from_array(self):
        input_value = np.array([[1.8, -1.1], [1.0, 7.3], [0.1, -1.0]])
        expected_value = [[0.85815, 0.24974], [0.73106, 0.99932], [0.52498, 0.26894]]

        result = np.round(self.function.value(input_data=input_value), 5)

        self.assertEqual(expected_value, result.tolist())

    def test_should_return_values_derived_from_array(self):
        input_value = np.array([[-100, 100], [0, 1], [-1, -0]])
        expected_value = [[0, 0], [0.25, 0.19661], [0.19661, 0.25]]

        result = np.round(self.function.derived(input_data=input_value), 5)

        self.assertEqual(expected_value, result.tolist())

    @pytest.mark.skipif(reason="never run")
    def test_show_draw(self):
        date_range = np.linspace(-10, 10).reshape([50, 1])
        data = self.function.value(input_data=date_range)
        data_derived = self.function.derived(input_data=date_range)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        axes[0].plot(date_range, data)
        axes[1].plot(date_range, data_derived)
        fig.tight_layout()
        plt.show()
