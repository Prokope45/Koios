"""GraphState.py

GraphState during agent workflow.

Author: Jared Paubel jpaubel@pm.me
version 0.1.0
"""
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """Represents the state of agent graph.

    Attributes:
        question (str): The question asked.
        generation (str): LLM generation.
        search_query (str): Revised question for web search.
        context (str): Web search context result.
    """
    question: str
    generation: str
    search_query: str
    context: str
