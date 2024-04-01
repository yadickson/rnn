class ActivationFunction:
    def value(self, input_data):
        raise NotImplementedError

    def derived(self, input_data):
        raise NotImplementedError
