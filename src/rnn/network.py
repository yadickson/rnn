from rnn.functions.loss_function import LossFunction


class Network:

    def __init__(self, layers, loss_function: LossFunction):
        self.layers = layers
        self.loss_function = loss_function

    def process(self, input_data):
        samples = len(input_data)
        result = []

        for i in range(samples):
            output = input_data[i]
            for li in range(len(self.layers)):
                output = self.layers[li].forward_propagation(output)
            result.append(output)

        return result

    def training(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)
        errors = []

        for i in range(epochs):
            err = 0
            for j in range(samples):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                err += self.loss_function.value(y_train[j], output)

                error = self.loss_function.derived(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            err /= samples

            errors.append(err)

            if err < learning_rate:
                break

        return errors
