"""api_server.py

FastAPI server for exposing the Koios RAG model via REST API.

Author: Jared Paubel jpaubel@pm.me
version 0.1.0
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

sys.path.insert(0, "/app")

from src.koios.AgentWorkflow.AgentWorkflow import AgentWorkflow
from src.koios.AgentPrompt.AgentPrompt import AgentPrompt
from src.config import Config

app = FastAPI(title="Koios RAG API", version="0.1.0")


class ChatMessage(BaseModel):
    """A single message in the conversation history.

    Attributes:
        role (str): Either ``"user"`` or ``"assistant"``.
        content (str): The message text.
    """
    role: str
    content: str


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str
    name: Optional[str] = "User"
    model: Optional[str] = None
    temperature: Optional[float] = 0.5
    enable_internet_search: Optional[bool] = None
    history: Optional[List[ChatMessage]] = []


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    query: str
    name: str
    generation: str
    model: str
    history: List[ChatMessage] = []


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker healthcheck."""
    return {"status": "healthy"}


@app.get("/models")
async def get_models():
    """Get available models from Ollama."""
    try:
        models = AgentPrompt.get_available_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a RAG query via POST request.

    For multi-turn conversation, take the `history` list contents and paste it
    into the request history list.
    
    Args:
        request: QueryRequest containing query, name, model, temperature, search, and history settings.
        
    Returns:
        QueryResponse with the generated answer.
    """
    try:
        config = Config()
        config.setup()
        
        # Use provided model or default to first available
        if request.model:
            selected_model = request.model
        else:
            model_options = AgentPrompt.get_available_models()
            selected_model = model_options[0] if model_options else "llama3.2"
        
        # Use provided internet search setting or config default
        enable_search = request.enable_internet_search if request.enable_internet_search is not None else config.enable_internet_search

        # Convert Pydantic ChatMessage objects to plain dicts so that
        # WorkflowActions._to_langchain_messages can process them uniformly
        # (same format used by the Streamlit UI).
        history_dicts = [{"role": m.role, "content": m.content} for m in (request.history or [])]

        # Create workflow and process query
        workflow = AgentWorkflow(selected_model, request.temperature, enable_internet_search=enable_search)
        output = workflow.local_agent.invoke({
            "question": request.query,
            "history": history_dicts,
            "context": "",
            "generation": "",
            "search_query": "",
        })

        generation = output.get("generation", "No generation produced.")

        # Build updated history to return to the client so they can pass it
        # back in subsequent requests for multi-turn conversation support.
        updated_history = history_dicts + [
            {"role": "user", "content": request.query},
            {"role": "assistant", "content": generation},
        ]

        return QueryResponse(
            query=request.query,
            name=request.name,
            generation=generation,
            model=selected_model,
            history=[ChatMessage(**m) for m in updated_history],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query")
async def process_query_stateless(query: str, name: str = "User", model: Optional[str] = None):
    """Process a RAG query via GET request (stateless, no history).

    For multi-turn conversations with history, use the POST ``/query``
    endpoint and pass the ``history`` field returned by each response back
    in the next request.

    Args:
        query: The research question.
        name: User name (optional).
        model: Model to use (optional, defaults to first available).

    Returns:
        QueryResponse with the generated answer.
    """
    request = QueryRequest(query=query, name=name, model=model)
    return await process_query(request)
