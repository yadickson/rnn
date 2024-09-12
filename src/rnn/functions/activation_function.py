from abc import abstractmethod
from abc import ABCMeta


class ActivationFunction(metaclass=ABCMeta):

    @abstractmethod
    def value(self, input_data):
        pass

    @abstractmethod
    def derived(self, input_data):
        pass
