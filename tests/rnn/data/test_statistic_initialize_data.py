from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from faker import Faker

from rnn.data.statistic_data import StatisticData
from rnn.data.statistic_initialize_data import StatisticInitializeData


class TestStatisticInitializeData(TestCase):

    def setUp(self):
        self.faker = Faker()

        self.input_size = self.faker.random.randint(20, 30)
        self.output_size = self.faker.random.randint(40, 50)
        self.generator_stub = MagicMock(StatisticData)

        self.initializer = StatisticInitializeData(
            input_size=self.input_size, output_size=self.output_size, generator=self.generator_stub
        )

    def test_should_check_input_size_is_assigned(self):
        self.assertEqual(self.input_size, self.initializer.input_size)

    def test_should_check_output_is_assigned(self):
        self.assertEqual(self.output_size, self.initializer.output_size)

    def test_should_check_generator_is_assigned(self):
        self.assertEqual(self.generator_stub, self.initializer.generator)

    def test_should_check_generator_was_called_two_times(self):
        self.initializer.get_next_trained_data()

        self.assertEqual(2, self.generator_stub.create.call_count)

        calls = self.generator_stub.method_calls

        self.assertTrue(calls[0] == ("create", {"input_size": self.input_size, "output_size": self.output_size}))
        self.assertTrue(calls[1] == ("create", {"input_size": 1, "output_size": self.output_size}))

    def test_should_return_trained_values(self):

        weights = np.random.rand(20, 30)
        bias = np.random.rand(10, 20)

        values = [weights, bias]

        self.generator_stub.create.side_effect = values

        result = self.initializer.get_next_trained_data()

        self.assertEqual(weights.tolist(), result.weights.tolist())
        self.assertEqual(bias.tolist(), result.bias.tolist())
