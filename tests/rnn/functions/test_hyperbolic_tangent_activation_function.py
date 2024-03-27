from unittest import TestCase
from rnn.functions.hyperbolic_tangent_activation_function import (
    HyperbolicTangentActivationFunction,
)
from faker import Faker
import matplotlib.pyplot as plt
import numpy as np
import pytest


class TestHyperbolicTangentActivationFunction(TestCase):

    def setUp(self):
        self.faker = Faker()
        self.function = HyperbolicTangentActivationFunction()

    def test_should_return_the_same_value_when_input_value_is_greater_than_zero(self):
        input_value = self.faker.random.randint(100, 200)
        self.assertEqual(1.0, self.function.value(input_value))

    def test_should_return_zero_value_when_input_is_zero(self):
        self.assertEqual(0, self.function.value(0))

    def test_should_return_minus_one_value_when_input_value_is_less_than_zero(self):
        input_value = self.faker.random.randint(-200, -100)
        self.assertEqual(-1.0, self.function.value(input_value))

    def test_should_return_zero_derived_value_when_input_value_is_greater_than_zero(
        self,
    ):
        input_value = self.faker.random.randint(100, 200)
        self.assertEqual(0, self.function.derived(input_value))

    def test_should_return_zero_derived_value_when_input_value_is_zero(self):
        self.assertEqual(1, self.function.derived(0))

    def test_should_return_zero_derived_value_when_input_value_is_less_than_zero(self):
        input_value = self.faker.random.randint(-200, -100)
        self.assertEqual(0, self.function.derived(input_value))

    @pytest.mark.skipif(reason="never run")
    def test_show_draw(self):
        rango = np.linspace(-10, 10).reshape([50, 1])
        datos_sigmoide = self.function.value(rango)
        datos_sigmoide_derivada = self.function.derived(rango)

        # Cremos los graficos
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        axes[0].plot(rango, datos_sigmoide)
        axes[1].plot(rango, datos_sigmoide_derivada)
        fig.tight_layout()
        plt.show()
