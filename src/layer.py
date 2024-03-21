import numpy as np
from scipy import stats

class Layer:
  def __init__(self, number_of_neurons = 1, activation_function = None, previous_layer = Layer(), round = 3):
    self.number_of_neurons = number_of_neurons
    self.activation_function = activation_function
    self.previous_number_of_neurons = previous_layer.number_of_neurons
    self.bias = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size= self.number_of_neurons).reshape(1,self.number_of_neurons),round)
    self.weights = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size= self.number_of_neurons * self.previous_number_of_neurons).reshape(self.previous_number_of_neurons,self.number_of_neurons),round)
