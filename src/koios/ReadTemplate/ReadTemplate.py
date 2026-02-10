"""ReadTemplate.py

Read template file and return output to be used as llm prompt.

Author: Jared Paubel jpaubel@pm.me
version 0.1.0
"""
from src.koios.enums.Template import Template


class ReadTemplate:
    """ReadPrompt singleton for reading template files."""

    _instance = None

    def __init__(self) -> None:
        """Deny instantiation of class."""
        return None

    def __new__(cls):
        """Instantiates singleton if none exist yet.

        Returns:
            cls: Class to check if instance exists.
        """
        if cls._instance is None:
            cls._instance = super(ReadTemplate, cls).__new__(cls)
        return cls._instance

    def get_contents(self, template: Template) -> str:
        """Public method to get contents of template file.

        Args:
            template (Template): Template enum containing path.

        Returns:
            str: Contents of template file.
        """
        return self.__get_contents(template)

    def __get_contents(self, template: Template) -> str:
        """Private method to get contents of template file.

        Args:
            template (Template): Template enum containing path.

        Returns:
            str: Contents of template file.
        """
        output = []
        with open(template.path, 'r') as reader:
            output.extend(reader.readlines())
        return "".join(output)
