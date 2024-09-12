from typing import Any

import numpy as np
from scipy import stats


class StatisticData:

    @staticmethod
    def create(input_size: int, output_size: int) -> Any:
        random = stats.truncnorm.rvs(-1, 1, size=input_size * output_size)
        return np.round(random.reshape(input_size, output_size), 3)
