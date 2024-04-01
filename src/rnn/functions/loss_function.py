class LossFunction:
    def value(self, real_value, calculated_value):
        raise NotImplementedError

    def derived(self, real_value, calculated_value):
        raise NotImplementedError
