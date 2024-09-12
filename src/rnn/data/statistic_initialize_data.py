from rnn.data.initialize_data import InitializeData
from rnn.data.statistic_data import StatisticData
from rnn.data.trained_data import TrainedData


class StatisticInitializeData(InitializeData):

    def __init__(self, input_size: int, output_size: int, generator: StatisticData = StatisticData()) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.generator = generator

    def get_next_trained_data(self) -> TrainedData:
        weights = self.generator.create(input_size=self.input_size, output_size=self.output_size)
        bias = self.generator.create(input_size=1, output_size=self.output_size)
        return TrainedData(weights, bias)
