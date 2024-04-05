from unittest import TestCase

from rnn.layers.layer import Layer


class TestLayer(TestCase):

    def setUp(self):
        self.layer = Layer()

    def test_should_check_input_is_assigned_with_none(self):
        self.assertEqual(None, self.layer.input)

    def test_should_check_output_is_assigned_with_none(self):
        self.assertEqual(None, self.layer.output)

    def test_should_throws_not_implement_exception_when_method_forward_propagation_is_called(self):
        self.assertRaises(NotImplementedError, self.layer.forward_propagation, None)

    def test_should_throws_not_implement_exception_when_method_backward_propagation_is_called(self):
        self.assertRaises(NotImplementedError, self.layer.backward_propagation, None, None)

    def test_should_throws_not_implement_exception_when_method_get_trained_values_is_called(self):
        self.assertRaises(NotImplementedError, self.layer.get_trained_values)
