from unittest import TestCase

from rnn.functions.activation_function import ActivationFunction


class TestActivationFunction(TestCase):

    def setUp(self):
        self.function = ActivationFunction()

    def test_should_throws_not_implement_exception_when_method_value_is_called(self):
        self.assertRaises(NotImplementedError, self.function.value, None)

    def test_should_throws_not_implement_exception_when_method_derived_is_called(self):
        self.assertRaises(NotImplementedError, self.function.derived, None)
