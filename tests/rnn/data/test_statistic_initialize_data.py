from unittest import TestCase

import faker.generator
from faker import Faker

from rnn.data.statistic_initialize_data import StatisticInitializeData


class TestStatisticInitializeData(TestCase):

    def setUp(self):
        self.faker = Faker()
        self.initializer = StatisticInitializeData()

    def test_should_return_empty_array_when_method_create_is_called_with_both_params_zero(
        self,
    ):
        result = self.initializer.create(input_size=0, output_size=0)
        self.assertEqual(0, result.size)

    def test_should_return_array_with_one_element_when_method_create_is_called_with_both_params_one(
        self,
    ):
        result = self.initializer.create(input_size=1, output_size=1)
        self.assertEqual(1, result.size)

    def test_should_return_array_with_one_element_and_greater_that_minus_one(self):
        result = self.initializer.create(input_size=1, output_size=1)
        self.assertGreaterEqual(1, result[0][-1])

    def test_should_return_array_with_one_element_and_less_that_one(self):
        result = self.initializer.create(input_size=1, output_size=1)
        self.assertLessEqual(result[0][-1], 1)

    def test_should_return_array_when_method_create_is_called_with_input_and_output_values(
        self,
    ):
        input_size = faker.generator.random.randint(10, 20)
        output_size = faker.generator.random.randint(30, 40)
        result = self.initializer.create(input_size=input_size, output_size=output_size)
        self.assertEqual(input_size * output_size, result.size)
