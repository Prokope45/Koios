"""models.py

Response and request models used in the API endpoints.

Author: Jared Paubel jpaubel@pm.me
version 0.1.0
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class ChatMessage(BaseModel):
    """A single message in the conversation history.

    Attributes:
        role (str): Either `"user"` or `"assistant"`.
        content (str): The message text.
    """
    role: str
    content: str


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="Message to query AI.")
    model: Optional[str] = Field(
        None,
        description=(
            "Model to query. Will pick the first available "
            "model if no model name is provided."
        )
    )
    temperature: Optional[float] = Field(0.5, description="Injected randomness into model")
    enable_internet_search: Optional[bool] = Field(False, description="Allow model to query the internet for context.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What is machine learning?",
                    "model": None,
                    "temperature": 0.7,
                    "enable_internet_search": False
                }
            ]
        }
    }


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    query: str
    user_id: str
    generation: str
    model: str
    history: List[ChatMessage] = []


class HistoryResponse(BaseModel):
    """Response model for history endpoints."""
    user_id: str
    message_count: int
    history: List[ChatMessage] = []


class ClearHistoryResponse(BaseModel):
    """Response model for the DELETE /history endpoint."""
    user_id: str
    messages_deleted: int
