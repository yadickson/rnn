from abc import ABCMeta, abstractmethod

from rnn.data.trained_data import TrainedData


class InitializeData(metaclass=ABCMeta):

    @abstractmethod
    def get_next_trained_data(self) -> TrainedData:
        pass
