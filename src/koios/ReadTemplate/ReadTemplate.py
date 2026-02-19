"""ReadTemplate.py

Read template file and return output to be used as llm prompt.

Author: Jared Paubel jpaubel@pm.me
version 0.1.0
"""
import os
import re
from transformers import AutoTokenizer

from src.koios.enums.Template import Template


class ReadTemplate:
    """ReadPrompt singleton for reading template files."""

    _instance = None

    # Maps model name patterns (regex) to HuggingFace tokenizer IDs.
    # Ungated mirrors are preferred so no HF_TOKEN is required.
    _TOKENIZER_MAP: dict[str, str] = {
        r"llama":           "unsloth/Llama-3.2-1B-Instruct",
        r"mistral|mixtral": "mistralai/Mistral-Nemo-Instruct-2407",
    }


    # Cache loaded tokenizers so each is only downloaded/initialised once.
    _tokenizer_cache: dict[str, AutoTokenizer] = {}

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
        """Public method to get raw contents of template file.

        Args:
            template (Template): Template enum containing path.

        Returns:
            str: Contents of template file.
        """
        return self.__get_contents(template)

    def get_chat_prompt(self, model: str, template: Template) -> str:
        """Return a fully tokenized chat prompt string for the given model.

        Reads the plain-text template file, splits it on the ``---`` separator
        into a system block and an optional user block, then applies the
        HuggingFace chat template for *model* so that the correct
        model-specific special tokens are injected automatically.

        LangChain ``{placeholder}`` variables (e.g. ``{question}``) are left
        untouched because Jinja2 (used by ``apply_chat_template``) uses
        ``{{ }}`` syntax and ignores single-brace placeholders. LangChain's
        ``PromptTemplate`` handles variable substitution at invoke time.

        The ``---`` separator is optional: if absent the entire file is treated
        as the system message with no separate user turn.

        Args:
            model (str): The model identifier (e.g. ``"llama3.2"``).
            template (Template): Template enum selecting the ``.txt`` file.

        Returns:
            str: Fully formatted prompt string with model-specific tokens,
                ready to be passed to ``PromptTemplate``.
        """
        raw = self.__get_contents(template)
        parts = raw.split("---", 1)

        system_content = parts[0].strip()
        user_content = parts[1].strip() if len(parts) > 1 else None

        messages = [{"role": "system", "content": system_content}]
        if user_content:
            messages.append({"role": "user", "content": user_content})

        tokenizer = self.__get_tokenizer(model)
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

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

    def __get_tokenizer(self, model: str) -> AutoTokenizer:
        """Resolve and cache the HuggingFace tokenizer for *model*.

        Args:
            model (str): Model identifier string (case-insensitive match).

        Returns:
            AutoTokenizer: Loaded tokenizer for the matched model family.

        Raises:
            ValueError: If no tokenizer mapping exists for *model*.
        """
        model_lower = model.lower()
        for pattern, hf_id in self._TOKENIZER_MAP.items():
            if re.search(pattern, model_lower):
                if hf_id not in self._tokenizer_cache:
                    token = os.getenv("HF_TOKEN")
                    self._tokenizer_cache[hf_id] = AutoTokenizer.from_pretrained(hf_id, token=token)
                return self._tokenizer_cache[hf_id]
        raise ValueError(
            f"No tokenizer mapping found for model '{model}'. "
            f"Add an entry to ReadTemplate._TOKENIZER_MAP."
        )
