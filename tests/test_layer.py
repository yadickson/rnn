from src.layer import Layer


class TestLayer:
  def test_should_return_one_number_of_neurons_when_create_layer_with_empty_parameters(self):
    layer = Layer()
    assert 1 == layer.number_of_neurons
