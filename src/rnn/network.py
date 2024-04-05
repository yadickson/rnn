import logging
from typing import List

import numpy as np

from rnn.data.training_data import TrainingData
from rnn.functions.loss_function import LossFunction


class Network:

    def __init__(self, layers, loss_function: LossFunction):
        self.layers = layers
        self.loss_function = loss_function

    def process(self, input_data):
        output = input_data

        for layer in self.layers:
            output = layer.forward_propagation(input_data=output)

        return output

    def training(self, input_training, output_training, learning: List[TrainingData]):
        errors = []
        samples = len(input_training)

        for li in range(len(learning)):

            for i in range(learning[li].epochs):
                error = 0
                for si in range(samples):
                    output = np.array(self.process(input_data=input_training[si]))

                    error = self.loss_function.value(output_training[si], output)
                    error_training = self.loss_function.derived(output_training[si], output)

                    for layer in reversed(self.layers):
                        error_training = layer.backward_propagation(error_training, learning[li].learning_rate)

                    logging.info(
                        "learning %d/%d: learning_rate %f epoch %d/%d sample[%d] error=%f"
                        % (li + 1, len(learning), learning[li].learning_rate, i + 1, learning[li].epochs, si, error)
                    )

                errors.append(error)

        return errors
