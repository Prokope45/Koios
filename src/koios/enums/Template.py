"""TemplatePath.py

Template path of agent templates.

Author: Jared Paubel jpaubel@pm.me
version 0.1.0
"""
from enum import Enum
from pathlib import Path


class Template(Enum):
    GENERATE = "generate_template"
    ROUTER = "router_template"
    QUERY = "query_template"

    def __init__(self, value: str):
        super().__init__(value)
        src_dir_path = Path(__file__).parent.parent
        self.__target_dir_path = src_dir_path.joinpath("ReadTemplate")

    @property
    def path(self) -> str:
        return "{}/templates/{}.txt".format(self.__target_dir_path, self.value)
