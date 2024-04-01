import numpy as np
from numpy import ndarray
from scipy import stats

from rnn.data.initialize_data import InitializeData


class StatisticInitializeData(InitializeData):
    def create(self, input_size, output_size) -> ndarray:
        random = stats.truncnorm.rvs(-1, 1, size=input_size * output_size)
        return np.round(random.reshape(input_size, output_size), 3)
