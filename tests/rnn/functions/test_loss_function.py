from unittest import TestCase

from rnn.functions.loss_function import LossFunction


class TestLossFunction(TestCase):

    def setUp(self):
        self.function = LossFunction()

    def test_should_throws_not_implement_exception_when_method_value_is_called(self):
        self.assertRaises(NotImplementedError, self.function.value, None, None)

    def test_should_throws_not_implement_exception_when_method_derived_is_called(self):
        self.assertRaises(NotImplementedError, self.function.derived, None, None)
