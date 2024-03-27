class ActivationFunction:
    def value(self, input_value):
        raise NotImplementedError

    def derived(self, input_data):
        raise NotImplementedError
