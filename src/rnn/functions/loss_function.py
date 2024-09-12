from abc import ABCMeta, abstractmethod
from typing import Any


class LossFunction(metaclass=ABCMeta):

    @abstractmethod
    def value(self, real_value: Any, calculated_value: Any) -> Any:
        pass

    @abstractmethod
    def derived(self, real_value: Any, calculated_value: Any) -> Any:
        pass
