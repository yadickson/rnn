from unittest import TestCase

import numpy as np
from faker import Faker

from rnn.functions.mean_squared_error_loss_function import \
    MeanSquaredErrorLossFunction


class TestMeanSquaredErrorLossFunction(TestCase):

    def setUp(self):
        self.faker = Faker()
        self.function = MeanSquaredErrorLossFunction()

    def test_should_return_zero_value_when_real_value_and_calculate_value_are_equal(self):
        value = self.faker.random.randint(10, 20)
        self.assertEqual(0, self.function.value(value, value))

    def test_should_return_mse_value_when_real_value_and_calculate_value_are_not_equal(self):
        real_value = self.faker.random.randint(17, 19)
        calculated_value = self.faker.random.randint(20, 22)
        self.assertNotEqual(0, self.function.value(real_value, calculated_value))

    def test_should_return_zero_value_derived_when_real_value_and_calculate_value_are_equal(self):
        value = self.faker.random.randint(10, 20)
        self.assertEqual(0, self.function.derived(value, value))

    def test_should_return_mse_value_derived_when_real_value_and_calculate_value_are_not_equal(self):
        real_value = self.faker.random.randint(17, 19)
        calculated_value = self.faker.random.randint(20, 22)
        self.assertNotEqual(0, self.function.derived(real_value, calculated_value))

    def test_should_return_values_from_array_when_both_are_equals(self):
        real_value = np.array([1.8, -1.1, 0.1, -1.0])
        calculated_value = np.array([1.8, -1.1, 0.1, -1.0])

        result = self.function.value(real_value, calculated_value)

        self.assertEqual(0, result)

    def test_should_return_values_from_array(self):
        real_value = np.array([1.8, -1.1, 0.1, -1.0])
        calculated_value = np.array([1.8, 0.0, 0.1, 0.0])

        result = self.function.value(real_value, calculated_value)

        self.assertEqual(0.5525, result.tolist())

    def test_should_return_values_derived_from_array_when_both_are_equals(self):
        real_value = np.array([1.8, -1.1, 0.1, -1.0])
        calculated_value = np.array([1.8, -1.1, 0.1, -1.0])

        result = self.function.derived(real_value, calculated_value)

        self.assertEqual([0, 0, 0, 0], result.tolist())

    def test_should_return_values_derived_from_array(self):
        real_value = np.array([1.8, -1.1, 0.1, -1.0])
        calculated_value = np.array([1.8, 0.0, 0.2, 0.0])

        result = self.function.derived(real_value, calculated_value)

        self.assertEqual([0.0, 1.1, 0.1, 1.0], result.tolist())
