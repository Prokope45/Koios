"""api_server.py

FastAPI server for exposing the Koios RAG model via REST API.

Author: Jared Paubel jpaubel@pm.me
version 0.2.0
"""
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from typing import List, Optional, Annotated
from jose import JWTError, jwt
import sys
import os

sys.path.insert(0, "/app")

from src.koios.AgentWorkflow.AgentWorkflow import AgentWorkflow
from src.koios.AgentPrompt.AgentPrompt import AgentPrompt
from src.koios.ChatHistoryStore import ChatHistoryStore
from src.config import Config, logger

app = FastAPI(title="Koios RAG API", version="0.2.0")

# ---------------------------------------------------------------------------
# Shared singletons
# ---------------------------------------------------------------------------
# Config is loaded once at startup so that environment variables are available
# to the ChatHistoryStore before any request arrives.
_config = Config()
_config.setup()

# Single ChatHistoryStore instance shared across all requests (thread-safe via
# SQLAlchemy's connection pool and per-session context managers).
_history_store: ChatHistoryStore = ChatHistoryStore(_config.chat_history_db_path)

# HTTPBearer scheme used to extract the JWT from the Authorization header.
_bearer_scheme = HTTPBearer()


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


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    query: str
    name: str
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


def verify_jwt_token(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> dict:
    """FastAPI dependency that validates the JWT Bearer token.

    Extracts and decodes the JWT from the ``Authorization: Bearer <token>``
    header.  The token is verified against ``KOIOS_JWT_SECRET_KEY`` using the
    algorithm specified by ``KOIOS_JWT_ALGORITHM`` (default: ``HS256``).

    When ``KOIOS_JWT_EXPIRY_HOURS`` is **not** set, the ``exp`` claim is not
    enforced (tokens are accepted regardless of expiry).  When it is set,
    expired tokens are rejected.

    Args:
        credentials: Bearer credentials extracted by FastAPI's HTTPBearer scheme.

    Returns:
        dict: The decoded JWT payload.

    Raises:
        HTTPException 401: If the secret key is not configured, the token is
            malformed, or (when expiry is enforced) the token has expired.
    """
    secret_key = _config.jwt_secret_key
    if not secret_key:
        raise HTTPException(
            status_code=401,
            detail=(
                "JWT authentication is not configured. "
                "Set KOIOS_JWT_SECRET_KEY in the environment."
            ),
        )

    # When expiry is not configured we instruct python-jose to skip the `exp`
    # claim check entirely by passing options={"verify_exp": False}.
    decode_options = {}
    if _config.jwt_expiry_hours is None:
        decode_options["verify_exp"] = False

    try:
        payload = jwt.decode(
            credentials.credentials,
            secret_key,
            algorithms=[_config.jwt_algorithm],
            options=decode_options,
        )
    except JWTError as exc:
        logger.warning("JWT validation failed: %s", exc)
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token.",
        )

    return payload


def get_current_user(
    x_user_id: Annotated[Optional[str], Header()] = None,
    _token_payload: dict = Depends(verify_jwt_token),
) -> str:
    """FastAPI dependency that validates the ``X-User-ID`` request header.

    JWT authentication is performed first via :func:`verify_jwt_token`.  Only
    after the token is accepted is the ``X-User-ID`` header checked against
    the comma-separated list of approved identifiers stored in the
    ``KOIOS_APPROVED_USER_IDS`` environment variable.

    Args:
        x_user_id: Value of the ``X-User-ID`` HTTP header.
        _token_payload: Decoded JWT payload injected by :func:`verify_jwt_token`
            (not used directly; present to enforce JWT validation first).

    Returns:
        str: The validated user identifier.

    Raises:
        HTTPException 401: If the JWT is invalid, the header is missing, or
            the identifier is not in the approved list.
    """
    if not x_user_id:
        raise HTTPException(
            status_code=401,
            detail="Missing required header: X-User-ID",
        )

    approved = _config.approved_user_ids
    if not approved:
        raise HTTPException(
            status_code=401,
            detail=(
                "No approved users configured. "
                "Set KOIOS_APPROVED_USER_IDS in the environment."
            ),
        )

    if x_user_id not in approved:
        logger.warning("Rejected request from unknown user '%s'", x_user_id)
        raise HTTPException(
            status_code=401,
            detail=f"User ID '{x_user_id}' is not authorised.",
        )

    return x_user_id


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker healthcheck."""
    return {"status": "healthy"}


@app.get("/models")
async def get_models(
    _token_payload: dict = Depends(verify_jwt_token),
):
    """Get available models from Ollama.

    Requires a valid JWT Bearer token (``Authorization: Bearer <token>``).
    No ``X-User-ID`` header is needed for this endpoint.

    Args:
        _token_payload: Decoded JWT payload injected by :func:`verify_jwt_token`.

    Returns:
        dict: A mapping of ``"models"`` to the list of available model names.
    """
    try:
        models = AgentPrompt.get_available_models()
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    user_id: str = Depends(get_current_user),
):
    """Process a RAG query via POST request.

    Chat history is automatically loaded from the database for the
    authenticated user and passed to the history-aware retriever so that the
    model can contextualise the current question against prior turns.  The
    new user message and the generated response are persisted back to the
    database after each successful call.

    The ``X-User-ID`` header is **required**.  Only identifiers listed in
    ``KOIOS_APPROVED_USER_IDS`` are accepted.

    Args:
        request: QueryRequest containing query, name, model, temperature, and
            search settings.
        user_id: Validated user identifier injected by :func:`get_current_user`.

    Returns:
        QueryResponse with the generated answer and the full updated history.
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
        workflow = AgentWorkflow(
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

        return QueryResponse(
            query=request.query,
            name=request.name,
            generation=generation,
            model=selected_model,
            history=[ChatMessage(**m) for m in updated_history],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query")
async def process_query_stateless(
    query: str,
    name: str = "User",
    model: Optional[str] = None,
    user_id: str = Depends(get_current_user),
):
    """Process a RAG query via GET request (stateless, no history).

    This endpoint does **not** load or persist chat history.  For multi-turn
    conversations with persistent per-user history, use the POST ``/query``
    endpoint instead.

    The ``X-User-ID`` header is **required**.

    Args:
        query: The research question.
        name: User name (optional).
        model: Model to use (optional, defaults to first available).
        user_id: Validated user identifier injected by :func:`get_current_user`.

    Returns:
        QueryResponse with the generated answer and an empty history list.
    """
    try:
        config = Config()
        config.setup()

        if model:
            selected_model = model
        else:
            model_options = AgentPrompt.get_available_models()
            selected_model = model_options[0] if model_options else "llama3.2"

        enable_search = config.enable_internet_search

        workflow = AgentWorkflow(
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

        return QueryResponse(
            query=query,
            name=name,
            generation=generation,
            model=selected_model,
            history=[],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history", response_model=HistoryResponse)
async def get_history(user_id: str = Depends(get_current_user)):
    """Retrieve the authenticated user's full chat history.

    The ``X-User-ID`` header is **required**.

    Args:
        user_id: Validated user identifier injected by :func:`get_current_user`.

    Returns:
        HistoryResponse containing the user's stored messages.
    """
    try:
        history_dicts = _history_store.get_history(user_id)
        return HistoryResponse(
            user_id=user_id,
            message_count=len(history_dicts),
            history=[ChatMessage(**m) for m in history_dicts],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/history", response_model=ClearHistoryResponse)
async def clear_history(user_id: str = Depends(get_current_user)):
    """Delete all stored chat history for the authenticated user.

    The ``X-User-ID`` header is **required**.

    Args:
        user_id: Validated user identifier injected by :func:`get_current_user`.

    Returns:
        ClearHistoryResponse with the number of messages deleted.
    """
    try:
        deleted = _history_store.clear_history(user_id)
        return ClearHistoryResponse(user_id=user_id, messages_deleted=deleted)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
