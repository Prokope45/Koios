"""api_server.py

FastAPI server for exposing the Koios RAG model via REST API.

Author: Jared Paubel jpaubel@pm.me
version 0.1.0
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import sys
import os

sys.path.insert(0, "/app")

from src.koios.AgentWorkflow.AgentWorkflow import AgentWorkflow
from src.koios.AgentPrompt.AgentPrompt import AgentPrompt
from src.config import Config

app = FastAPI(title="Koios RAG API", version="0.1.0")


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str
    name: Optional[str] = "User"
    model: Optional[str] = None
    temperature: Optional[float] = 0.5
    enable_internet_search: Optional[bool] = None


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    query: str
    name: str
    generation: str
    model: str


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
    
    Args:
        request: QueryRequest containing query, name, model, temperature, and search settings.
        
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
        
        # Create workflow and process query
        workflow = AgentWorkflow(selected_model, request.temperature, enable_internet_search=enable_search)
        output = workflow.local_agent.invoke({"question": request.query})
        
        generation = output.get("generation", "No generation produced.")
        
        return QueryResponse(
            query=request.query,
            name=request.name,
            generation=generation,
            model=selected_model
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query")
async def process_query_get(query: str, name: str = "User", model: Optional[str] = None):
    """Process a RAG query via GET request.
    
    Args:
        query: The research question.
        name: User name (optional).
        model: Model to use (optional, defaults to first available).
        
    Returns:
        QueryResponse with the generated answer.
    """
    request = QueryRequest(query=query, name=name, model=model)
    return await process_query(request)
