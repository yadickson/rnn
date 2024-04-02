import json


class TrainedData:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def get_values(self):
        return {
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
        }

    def get_json_values(self):
        return json.dumps(self.get_values())
