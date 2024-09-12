import json
from typing import Any


class JsonStruct:
    def __init__(self, content: Any) -> None:
        self.__dict__.update(content)


class JsonFile:
    @staticmethod
    def read(file_name: str) -> Any:

        with open(file_name, "r") as openfile:
            content = json.load(openfile)

        return json.loads(json.dumps(content), object_hook=JsonStruct)

    @staticmethod
    def write(file_name: str, input_data: Any) -> None:
        json_object = json.dumps(input_data, indent=2)
        with open(file_name, "w") as outfile:
            outfile.write(json_object)
