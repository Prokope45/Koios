"""ToonSerializer.py

Thin wrapper around the `toon_format` Python package that provides a
convenient interface for encoding Python data structures to TOON
(Token-Oriented Object Notation) before passing them as context to LLMs.

TOON achieves ~30-60% token reduction vs JSON while maintaining or improving
LLM comprehension accuracy, making it ideal for RAG context payloads.

Reference: https://github.com/toon-format/toon-python  (TOON Spec v3.0)

Author: Jared Paubel jpaubel@pm.me
version 0.1.0
"""
from __future__ import annotations

from typing import Any

from toon_format import encode, EncodeOptions


class ToonSerializer:
    """Encode Python objects to TOON format for token-efficient LLM context.

    Wraps the official ``toon_format`` package and provides a stable,
    project-level API with graceful fallback to plain-text on failure.

    Usage::

        serializer = ToonSerializer()
        toon_str = serializer.encode(data)

    Or use the convenience class-method::

        toon_str = ToonSerializer.dumps(data)
    """

    def __init__(self, indent: int = 2, delimiter: str = ",") -> None:
        """Construct a ToonSerializer.

        Args:
            indent (int): Number of spaces per indentation level. Default 2.
            delimiter (str): Field delimiter for tabular/primitive arrays.
                Allowed values: ``","`` (default), ``"\\t"``, ``"|"``.
        """
        self._options: EncodeOptions = {
            "indent": indent,
            "delimiter": delimiter,
        }

    def encode(self, value: Any) -> str:
        """Encode *value* to a TOON-formatted string.

        Falls back to ``str(value)`` if encoding fails so that the calling
        workflow is never interrupted by a serialisation error.

        Args:
            value: Any JSON-serialisable Python object (dict, list, str,
                int, float, bool, None).

        Returns:
            str: TOON-encoded representation of *value*, or the plain-text
                fallback on error.
        """
        try:
            return encode(value, self._options)
        except Exception as exc:  # pragma: no cover
            print(f"[ToonSerializer] Encoding failed, falling back to str: {exc}")
            return str(value)

    @classmethod
    def dumps(cls, value: Any, indent: int = 2, delimiter: str = ",") -> str:
        """Convenience class-method: encode *value* and return the TOON string.

        Args:
            value: Python object to encode.
            indent (int): Spaces per indentation level.
            delimiter (str): Field delimiter.

        Returns:
            str: TOON-encoded string.
        """
        return cls(indent=indent, delimiter=delimiter).encode(value)
