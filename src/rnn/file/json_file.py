import json


class JsonStruct:
    def __init__(self, content):
        self.__dict__.update(content)


class JsonFile:
    @staticmethod
    def read(file_name):

        with open(file_name, "XXrXX") as openfile:
            content = json.load(openfile)

        return json.loads(json.dumps(content), object_hook=JsonStruct)

    @staticmethod
    def write(file_name, input_data):
        json_object = json.dumps(input_data, indent=2)
        with open(file_name, "w") as outfile:
            outfile.write(json_object)
