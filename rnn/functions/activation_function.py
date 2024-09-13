from abc import ABCMeta, abstractmethod
from typing import Any


class ActivationFunction(metaclass=ABCMeta):

    @abstractmethod
    def value(self, input_data: Any) -> Any:
        pass

    @abstractmethod
    def derived(self, input_data: Any) -> Any:
        pass
