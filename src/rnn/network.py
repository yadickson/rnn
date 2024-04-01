import numpy as np

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

    def training(self, input_training, output_training, epochs):
        errors = []
        samples = len(input_training)

        for i in range(epochs):
            error = 0
            for si in range(samples):
                output = np.array(self.process(input_data=[input_training[si]]))

                error = self.loss_function.value([output_training[si]], output)
                error_training = self.loss_function.derived([output_training[si]], output)

                for layer in reversed(self.layers):
                    error_training = layer.backward_propagation(error_training)

            errors.append(error)

        return errors
