from rnn.data.initialize_data import InitializeData
from rnn.data.trained_data import TrainedData


class MemoryInitializeData(InitializeData):

    def __init__(self, trained_list):
        self.trained_iterator = iter(tuple(trained_list))

    def get_next_trained_data(self) -> TrainedData:
        return next(self.trained_iterator)
