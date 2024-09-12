from abc import ABCMeta, abstractmethod


class ActivationFunction(metaclass=ABCMeta):

    @abstractmethod
    def value(self, input_data):
        pass

    @abstractmethod
    def derived(self, input_data):
        pass
