from unittest import TestCase
from rnn.layer import Layer


class TestLayer(TestCase):

    def setUp(self):
        self.layer = Layer()

    def test_should_throws_not_implement_exception_when_method_forward_propagation_is_called(self):
        self.assertRaises(NotImplementedError, self.layer.forward_propagation, None)

    def test_should_throws_not_implement_exception_when_method_backward_propagation_is_called(self):
        self.assertRaises(NotImplementedError, self.layer.backward_propagation, None, None)
