import os
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import pytest
from faker import Faker

from rnn.functions.relu_activation_function import ReluActivationFunction


class TestReluActivationFunction(TestCase):

    def setUp(self):
        self.faker = Faker()
        self.function = ReluActivationFunction()

    def test_should_return_the_same_value_when_input_value_is_greater_than_zero(self):
        input_value = self.faker.random.randint(10, 20)
        self.assertEqual(input_value, self.function.value(input_data=input_value))

    def test_should_return_one_value_when_input_is_one(self):
        self.assertEqual(1.0, self.function.value(input_data=1))

    def test_should_return_zero_value_when_input_is_zero(self):
        self.assertEqual(0, self.function.value(input_data=0))

    def test_should_return_zero_value_when_input_value_is_less_than_zero(self):
        input_value = self.faker.random.randint(-20, -10)
        self.assertEqual(0, self.function.value(input_data=input_value))

    def test_should_return_one_derived_value_when_input_value_is_greater_than_zero(self):
        input_value = self.faker.random.randint(10, 20)
        self.assertEqual(1.0, self.function.derived(input_data=input_value))

    def test_should_return_zero_derived_value_when_input_value_is_zero(self):
        self.assertEqual(0, self.function.derived(input_data=0))

    def test_should_return_zero_derived_value_when_input_value_is_less_than_zero(self):
        input_value = self.faker.random.randint(-20, -10)
        self.assertEqual(0, self.function.derived(input_data=input_value))

    def test_should_return_values_from_array(self):
        input_value = np.array([[1.8, -1.1], [1.0, 7.3], [0.1, -1.0]])
        expected_value = [[1.8, 0.0], [1.0, 7.3], [0.1, 0.0]]

        result = self.function.value(input_data=input_value)

        self.assertEqual(expected_value, result.tolist())

    def test_should_return_values_derived_from_array(self):
        input_value = np.array([[1.8, -1.1], [1.0, 7.3], [0.1, -1.0]])
        expected_value = [[1.0, 0.0], [1.0, 1.0], [1.0, 0.0]]

        result = self.function.derived(input_data=input_value)

        self.assertEqual(expected_value, result.tolist())

    @pytest.mark.skipif(os.environ.get("TRAINING_TEST") is None, reason="run only in training mode")
    def test_show_draw(self):
        date_range = np.linspace(-10, 10).reshape([50, 1])
        date = self.function.value(input_data=date_range)
        date_derived = self.function.derived(input_data=date_range)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        axes[0].plot(date_range, date)
        axes[1].plot(date_range, date_derived)
        fig.tight_layout()
        plt.show()
