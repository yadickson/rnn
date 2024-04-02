from rnn.data.trained_data import TrainedData


class InitializeData:
    def get_next_trained_data(self) -> TrainedData:
        raise NotImplementedError
