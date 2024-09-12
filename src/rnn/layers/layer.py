from abc import ABCMeta, abstractmethod


class Layer(metaclass=ABCMeta):
    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward_propagation(self, input_data):
        pass

    @abstractmethod
    def backward_propagation(self, output_error, learning_rate):
        pass

    @abstractmethod
    def get_trained_values(self):
        pass
