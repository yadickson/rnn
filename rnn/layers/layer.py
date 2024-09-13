from abc import ABCMeta, abstractmethod
from typing import Any


class Layer(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.input: Any = None
        self.output: Any = None

    @abstractmethod
    def forward_propagation(self, input_data: Any) -> Any:
        pass

    @abstractmethod
    def backward_propagation(self, output_error: Any, learning_rate: Any) -> Any:
        pass

    @abstractmethod
    def get_trained_values(self) -> Any:
        pass
