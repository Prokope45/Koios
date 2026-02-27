"""api_server.py

FastAPI server for exposing the Koios RAG model via REST API.

Author: Jared Paubel jpaubel@pm.me
version 0.2.0
"""
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from typing import Optional, Annotated, Union
from pydantic import BaseModel
from jose import JWTError, jwt
import sys

sys.path.insert(0, "/app")

from src.koios.agent import Workflow, Prompt
from src.koios.data_store.ChatHistoryStore import ChatHistoryStore
from src.koios.toon_serializer.ToonSerializer import ToonSerializer
import src.app.models as models
from src.config import config, logger
from src.app.encryption import Encryption


def _wrap_response(data: Union[BaseModel, dict]):
    """Helper to encrypt response data if encryption is enabled."""
    if config.enable_encryption:
        if isinstance(data, BaseModel):
            data_dict = data.model_dump()
        else:
            data_dict = data
        encrypted = Encryption.encrypt(data_dict)
        return models.EncryptedResponse(encrypted_data=encrypted)
    return data

app = FastAPI(title="Koios RAG API", version="0.2.0")

# MARK:- Shared singletons
# Config is loaded once at startup so that environment variables are available
# to the ChatHistoryStore before any request arrives.
# config.setup()

# Single ChatHistoryStore instance shared across all requests (thread-safe via
# SQLAlchemy's connection pool and per-session context managers).
_history_store: ChatHistoryStore = ChatHistoryStore(config.chat_history_db_path)

# HTTPBearer scheme used to extract the JWT from the Authorization header.
_bearer_scheme = HTTPBearer()


def verify_jwt_token(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> dict:
    """FastAPI dependency that validates the JWT Bearer token.

    Extracts and decodes the JWT from the `Authorization: Bearer <token>`
    header.  The token is verified against `JWT_SECRET_KEY` using the
    algorithm specified by `JWT_ALGORITHM` (default: `HS256`).

    When `JWT_EXPIRY_HOURS` is **not** set, the `exp` claim is not
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
    secret_key = config.jwt_secret_key
    if not secret_key:
        raise HTTPException(
            status_code=401,
            detail=(
                "JWT authentication is not configured. "
                "Set JWT_SECRET_KEY in the environment."
            ),
        )

    # When expiry is not configured we instruct python-jose to skip the `exp`
    # claim check entirely by passing options={"verify_exp": False}.
    decode_options = {}
    if config.jwt_expiry_hours is None:
        decode_options["verify_exp"] = False

    try:
        payload = jwt.decode(
            credentials.credentials,
            secret_key,
            algorithms=[config.jwt_algorithm],
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
    """FastAPI dependency that validates the `X-User-ID` request header.

    JWT authentication is performed first via :func:`verify_jwt_token`.  Only
    after the token is accepted is the `X-User-ID` header checked against
    the comma-separated list of approved identifiers stored in the
    `APPROVED_USER_IDS` environment variable.

    Args:
        x_user_id: Value of the `X-User-ID` HTTP header.
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


# MARK:- Get Token
@app.post("/token", response_model=models.TokenResponse)
async def get_token(
    request: Request,
    x_user_id: Annotated[Optional[str], Header()] = None,
):
    """Issue a JWT Bearer token for an authorised user from a trusted IP.

    Two checks are performed before a token is issued:

    1. **IP whitelist** - The client IP is resolved from the
       `X-Forwarded-For` header (first entry, for reverse-proxy deployments)
       or, when that header is absent, from the direct TCP connection.
       `127.0.0.1` and `::1` are always permitted for local development.
       Additional IPs are configured via `AUTHORIZED_TOKEN_IPS`.

    2. **User authorisation** - The `X-User-ID` header must be present and
       must appear in the `APPROVED_USER_IDS` list.

    The generated token contains the following claims:

    * `sub` - the validated user identifier
    * `iss` - issuer string (`JWT_ISSUER`, default `"koios-api"`)
    * `iat` - UTC timestamp of issuance
    * `exp` - expiry timestamp (only included when `JWT_EXPIRY_HOURS`
      is set)

    Args:
        request: The incoming FastAPI request (used to extract the client IP).
        x_user_id: Value of the `X-User-ID` HTTP header.

    Returns:
        TokenResponse: OAuth2-style response with `access_token` and
            `token_type` fields.

    Raises:
        HTTPException 401: If the client IP is not authorised, the user ID is
            missing or not approved, or the JWT secret key is not configured.
    """
    # 1. Resolve client IP (reverse-proxy aware)
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        # X-Forwarded-For may contain a comma-separated chain; the leftmost
        # entry is the original client IP.
        client_ip = forwarded_for.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else ""

    authorized_ips = config.authorized_token_ips
    if client_ip not in authorized_ips:
        logger.warning(
            "Token request denied for IP '%s' (not in authorised list)", client_ip
        )
        raise HTTPException(
            status_code=401,
            detail=f"IP address '{client_ip}' is not authorised to request a token.",
        )

    # 2. Validate user identity
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
        logger.warning(
            "Token request denied for unknown user '%s' from IP '%s'",
            x_user_id,
            client_ip,
        )
        raise HTTPException(
            status_code=401,
            detail=f"User ID '{x_user_id}' is not authorised.",
        )

    # 3. Ensure the secret key is configured before signing
    secret_key = config.jwt_secret_key
    if not secret_key:
        raise HTTPException(
            status_code=401,
            detail=(
                "JWT authentication is not configured. "
                "Set JWT_SECRET_KEY in the environment."
            ),
        )

    # 4. Build JWT payload and sign
    now = datetime.now(tz=timezone.utc)
    payload: dict = {
        "sub": x_user_id,
        "iss": config.jwt_issuer,
        "iat": now,
    }

    expiry_hours = config.jwt_expiry_hours
    if expiry_hours is not None:
        payload["exp"] = now + timedelta(hours=expiry_hours)

    token = jwt.encode(payload, secret_key, algorithm=config.jwt_algorithm)

    logger.info(
        "Issued JWT token for user '%s' from IP '%s'", x_user_id, client_ip
    )
    return models.TokenResponse(access_token=token)


# MARK:- Get Models
@app.get("/models", response_model=Union[dict, models.EncryptedResponse])
async def get_models(
    _token_payload: dict = Depends(verify_jwt_token),
):
    """Get available models from Ollama.

    Requires a valid JWT Bearer token (`Authorization: Bearer <token>`).
    No `X-User-ID` header is needed for this endpoint.

    Args:
        _token_payload: Decoded JWT payload injected by :func:`verify_jwt_token`.

    Returns:
        dict: A mapping of `"models"` to the list of available model names.
    """
    try:
        model_list = Prompt.get_available_models()
        return _wrap_response({"models": model_list})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# MARK:- Process Query
@app.post("/query", response_model=Union[models.QueryResponse, models.EncryptedResponse])
async def process_query(
    request: Union[models.QueryRequest, models.EncryptedRequest],
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
        request: QueryRequest or EncryptedRequest containing query details.
        user_id: Validated user identifier injected by :func:`get_current_user`.

    Returns:
        QueryResponse or EncryptedResponse with the generated answer.
    """
    try:
        # 1. Handle decryption if enabled
        if config.enable_encryption:
            if not isinstance(request, models.EncryptedRequest):
                raise HTTPException(
                    status_code=400,
                    detail="Encrypted request required when ENABLE_ENCRYPTION is True."
                )
            try:
                decrypted = Encryption.decrypt(request.encrypted_data)
                actual_request = models.QueryRequest(**decrypted)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Decryption failed: {e}")
        else:
            if not isinstance(request, models.QueryRequest):
                raise HTTPException(
                    status_code=400,
                    detail="Plain request required when ENABLE_ENCRYPTION is False."
                )
            actual_request = request

        # Use provided model or default to first available
        if actual_request.model and actual_request.model != "":
            selected_model = actual_request.model
        else:
            model_options = Prompt.get_available_models()
            selected_model = model_options[0] if model_options else "llama3.2"

        # Use provided internet search setting or config default
        enable_search = (
            actual_request.enable_internet_search
            if actual_request.enable_internet_search is not None
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
        workflow = Workflow(
            selected_model, actual_request.temperature, enable_internet_search=enable_search
        )
        output = workflow.local_agent.invoke({
            "question": actual_request.query,
            "history": history_dicts,
            "context": "",
            "generation": "",
            "search_query": "",
        })

        generation = output.get("generation", "No generation produced.")

        # Persist the new turn to the database.
        _history_store.add_messages(user_id, [
            {"role": "user", "content": actual_request.query},
            {"role": "assistant", "content": generation},
        ])

        # Return the full updated history so clients can display it.
        updated_history = history_dicts + [
            {"role": "user", "content": actual_request.query},
            {"role": "assistant", "content": generation},
        ]

        response_obj = models.QueryResponse(
            query=actual_request.query,
            user_id=user_id,
            generation=generation,
            model=selected_model,
            history=[models.ChatMessage(**m) for m in updated_history],
        )
        return _wrap_response(response_obj)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# MARK:- Process Query (Stateless)
@app.get("/query", response_model=Union[models.QueryResponse, models.EncryptedResponse])
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

        response_obj = models.QueryResponse(
            query=query,
            user_id=user_id,
            generation=generation,
            model=selected_model,
            history=[],
        )
        return _wrap_response(response_obj)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# MARK:- Get History
@app.get("/history", response_model=Union[models.HistoryResponse, models.EncryptedResponse])
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
        response_obj = models.HistoryResponse(
            user_id=user_id,
            message_count=len(history_dicts),
            history=[models.ChatMessage(**m) for m in history_dicts],
        )
        return _wrap_response(response_obj)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# MARK:- Delete History
@app.delete("/history", response_model=Union[models.ClearHistoryResponse, models.EncryptedResponse])
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
        response_obj = models.ClearHistoryResponse(user_id=user_id, messages_deleted=deleted)
        return _wrap_response(response_obj)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# MARK:- Process Analysis
@app.post("/analyze", response_model=Union[models.AnalyzeResponse, models.EncryptedResponse])
async def process_analysis(
    request: Union[models.AnalyzeRequest, models.EncryptedRequest],
    user_id: str = Depends(get_current_user),
):
    """Process an analysis query with provided details.

    This endpoint takes a prompt and a list of metrics/details, serializes
    the details into TOON format, and injects them into the model's context.

    Args:
        request: AnalyzeRequest or EncryptedRequest containing prompt and details.
        user_id: Validated user identifier injected by :func:`get_current_user`.

    Returns:
        AnalyzeResponse or EncryptedResponse with the generated answer.
    """
    try:
        # Handle decryption if enabled
        if config.enable_encryption:
            if not isinstance(request, models.EncryptedRequest):
                raise HTTPException(
                    status_code=400,
                    detail="Encrypted request required when ENABLE_ENCRYPTION is True."
                )
            try:
                decrypted = Encryption.decrypt(request.encrypted_data)
                actual_request = models.AnalyzeRequest(**decrypted)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Decryption failed: {e}")
        else:
            if not isinstance(request, models.AnalyzeRequest):
                raise HTTPException(
                    status_code=400,
                    detail="Plain request required when ENABLE_ENCRYPTION is False."
                )
            actual_request = request

        # Use provided model or default to first available
        if actual_request.model and actual_request.model != "":
            selected_model = actual_request.model
        else:
            model_options = Prompt.get_available_models()
            selected_model = model_options[0] if model_options else "llama3.2"

        # Serialize details to TOON format for token efficiency
        details_list = [d.model_dump() for d in actual_request.details]
        toon_context = ToonSerializer.dumps({"details": details_list})

        # Create workflow and process query.
        # We disable internet search for this specialized analysis endpoint.
        workflow = Workflow(
            selected_model,
            actual_request.temperature or 0.5,
            enable_internet_search=False
        )

        output = workflow.local_agent.invoke({
            "question": actual_request.prompt,
            "history": [],  # Stateless analysis
            "context": toon_context,
            "generation": "",
            "search_query": "",
        })

        generation = output.get("generation", "No generation produced.")

        response_obj = models.AnalyzeResponse(
            prompt=actual_request.prompt,
            user_id=user_id,
            generation=generation,
            model=selected_model,
            details=actual_request.details,
        )
        return _wrap_response(response_obj)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
