class Layer:
  def __init__(self, number_of_neurons = 1, activation_function = None, previous_layer = Layer()):
    self.number_of_neurons = number_of_neurons
    self.activation_function = activation_function
    self.previous_number_of_neurons = previous_layer.number_of_neurons
