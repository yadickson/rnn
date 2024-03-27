class LossFunction:
    def value(self, real_value, output_desired):
        raise NotImplementedError

    def derived(self, real_value, output_desired):
        raise NotImplementedError
