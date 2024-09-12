from abc import abstractmethod
from abc import ABCMeta


class LossFunction(metaclass=ABCMeta):

    @abstractmethod
    def value(self, real_value, calculated_value):
        pass

    @abstractmethod
    def derived(self, real_value, calculated_value):
        pass
