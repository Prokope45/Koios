"""api_server.py

FastAPI server for exposing the Koios RAG model via REST API.

Author: Jared Paubel jpaubel@pm.me
version 0.2.0
"""
from fastapi import FastAPI, HTTPException, Header, Depends, Body
from typing import Optional, Annotated
import sys

sys.path.insert(0, "/app")

from src.koios.agent import Workflow, Prompt
from src.koios.data_store.ChatHistoryStore import ChatHistoryStore
import src.app.models as models
from src.config import config, logger

app = FastAPI(title="Koios RAG API", version="0.2.0")

# MARK:- Shared singletons
# Config is loaded once at startup so that environment variables are available
# to the ChatHistoryStore before any request arrives.
# config.setup()

# Single ChatHistoryStore instance shared across all requests (thread-safe via
# SQLAlchemy's connection pool and per-session context managers).
_history_store: ChatHistoryStore = ChatHistoryStore(config.chat_history_db_path)


# MARK:- Get User
def get_current_user(
    x_user_id: Annotated[Optional[str], Header()] = None,
) -> str:
    """FastAPI dependency that validates the `X-User-ID` request header.

    The header value is checked against the comma-separated list of approved
    identifiers stored in the `APPROVED_USER_IDS` environment variable.

    Args:
        x_user_id: Value of the `X-User-ID` HTTP header.

    Returns:
        str: The validated user identifier.

    Raises:
        HTTPException 401: If the header is missing or the identifier is not
            in the approved list.
    """
    if not x_user_id:
        raise HTTPException(
            status_code=401,
            detail="Missing required header: X-User-ID",
        )

    approved = config.approved_user_ids
    if not approved:
        raise HTTPException(
            status_code=401,
            detail=(
                "No approved users configured. "
                "Set APPROVED_USER_IDS in the environment."
            ),
        )

    if x_user_id not in approved:
        logger.warning("Rejected request from unknown user '%s'", x_user_id)
        raise HTTPException(
            status_code=401,
            detail=f"User ID '{x_user_id}' is not authorised.",
        )

    return x_user_id


# MARK:- HealthCheck
@app.get("/health")
async def health_check():
    """Health check endpoint for Docker healthcheck."""
    return {"status": "healthy"}


# MARK:- Get Models
@app.get("/models")
async def get_models():
    """Get available models from Ollama."""
    try:
        models = Prompt.get_available_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# MARK:- Process Query
@app.post("/query", response_model=models.QueryResponse)
async def process_query(
    request: models.QueryRequest,
    user_id: str = Depends(get_current_user),
):
    """Process a RAG query via POST request.

    Chat history is automatically loaded from the database for the
    authenticated user and passed to the history-aware retriever so that the
    model can contextualise the current question against prior turns.  The
    new user message and the generated response are persisted back to the
    database after each successful call.

    The `X-User-ID` header is **required**.  Only identifiers listed in
    `APPROVED_USER_IDS` are accepted.

    Args:
        request: QueryRequest containing query, user_id, model, temperature, and
            search settings.
        user_id: Validated user identifier injected by :func:`get_current_user`.

    Returns:
        QueryResponse with the generated answer and the full updated history.
    """
    try:
        # config.setup()

        # Use provided model or default to first available
        if request.model and request.model != "":
            selected_model = request.model
        else:
            model_options = Prompt.get_available_models()
            selected_model = model_options[0] if model_options else "llama3.2"

        # Use provided internet search setting or config default
        enable_search = (
            request.enable_internet_search
            if request.enable_internet_search is not None
            else config.enable_internet_search
        )

        # Load the user's persisted chat history from the database.
        history_dicts = _history_store.get_history(user_id)
        logger.info(
            "Loaded %d history message(s) for user '%s'",
            len(history_dicts),
            user_id,
        )

        # Create workflow and process query.
        # The history-aware retriever inside WorkflowActions.doc_search()
        # will reformulate the question using the full chat history so that
        # only the most contextually relevant documents are retrieved.
        workflow = Workflow(
            selected_model, request.temperature, enable_internet_search=enable_search
        )
        output = workflow.local_agent.invoke({
            "question": request.query,
            "history": history_dicts,
            "context": "",
            "generation": "",
            "search_query": "",
        })

        generation = output.get("generation", "No generation produced.")

        # Persist the new turn to the database.
        _history_store.add_messages(user_id, [
            {"role": "user", "content": request.query},
            {"role": "assistant", "content": generation},
        ])

        # Return the full updated history so clients can display it.
        updated_history = history_dicts + [
            {"role": "user", "content": request.query},
            {"role": "assistant", "content": generation},
        ]

        return models.QueryResponse(
            query=request.query,
            user_id=user_id,
            generation=generation,
            model=selected_model,
            history=[models.ChatMessage(**m) for m in updated_history],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# MARK:- Process Query (Stateless)
@app.get("/query")
async def process_query_stateless(
    query: str,
    model: Optional[str] = None,
    user_id: str = Depends(get_current_user),
):
    """Process a RAG query via GET request (stateless, no history).

    This endpoint does **not** load or persist chat history.  For multi-turn
    conversations with persistent per-user history, use the POST `/query`
    endpoint instead.

    The `X-User-ID` header is **required**.

    Args:
        query: The research question.
        user_id: User ID.
        model: Model to use (optional, defaults to first available).
        user_id: Validated user identifier injected by :func:`get_current_user`.

    Returns:
        QueryResponse with the generated answer and an empty history list.
    """
    try:
        # config.setup()

        if model:
            selected_model = model
        else:
            model_options = Prompt.get_available_models()
            selected_model = model_options[0] if model_options else "llama3.2"

        enable_search = config.enable_internet_search

        workflow = Workflow(
            selected_model, 0.5, enable_internet_search=enable_search
        )
        output = workflow.local_agent.invoke({
            "question": query,
            "history": [],
            "context": "",
            "generation": "",
            "search_query": "",
        })

        generation = output.get("generation", "No generation produced.")

        return models.QueryResponse(
            query=query,
            user_id=user_id,
            generation=generation,
            model=selected_model,
            history=[],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# MARK:- Get History
@app.get("/history", response_model=models.HistoryResponse)
async def get_history(user_id: str = Depends(get_current_user)):
    """Retrieve the authenticated user's full chat history.

    The `X-User-ID` header is **required**.

    Args:
        user_id: Validated user identifier injected by :func:`get_current_user`.

    Returns:
        HistoryResponse containing the user's stored messages.
    """
    try:
        history_dicts = _history_store.get_history(user_id)
        return models.HistoryResponse(
            user_id=user_id,
            message_count=len(history_dicts),
            history=[models.ChatMessage(**m) for m in history_dicts],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# MARK:- Delete History
@app.delete("/history", response_model=models.ClearHistoryResponse)
async def clear_history(user_id: str = Depends(get_current_user)):
    """Delete all stored chat history for the authenticated user.

    The `X-User-ID` header is **required**.

    Args:
        user_id: Validated user identifier injected by :func:`get_current_user`.

    Returns:
        ClearHistoryResponse with the number of messages deleted.
    """
    try:
        deleted = _history_store.clear_history(user_id)
        return models.ClearHistoryResponse(user_id=user_id, messages_deleted=deleted)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
