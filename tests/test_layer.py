import unittest
import layer


class TestLayer(unittest.TestCase):
  def test_should_return_one_number_of_neurons_when_create_layer_with_empty_parameters(self):
    object = layer.Layer()
    assert 1 == object.number_of_neurons
