from rnn.data.trained_data import TrainedData
from abc import abstractmethod
from abc import ABCMeta


class InitializeData(metaclass=ABCMeta):

    @abstractmethod
    def get_next_trained_data(self) -> TrainedData:
        pass
