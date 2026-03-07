from fastapi import HTTPException, Header, Depends
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.security import HTTPBearer
from typing import Optional, Annotated
from jose import JWTError, jwt

from src.config import config, logger
from src.app.encryption import Encryption


class Auth:
    """Authentication utility class for JWT-based API authentication.

    This class provides methods for:
    - IP address authorization checks
    - JWT token verification and validation
    - User identity extraction from requests
    - User authorization validation

    Usage in FastAPI routes:

        from src.app.api.auth import Auth

        @app.get("/protected")
        async def protected_route(
            current_user: str = Depends(Auth.get_current_user),
        ):
            return {"user": current_user}

        @app.post("/token")
        async def get_token(
            client_ip: str = Depends(Auth.check_ip_authorization),
            payload: dict = Depends(Auth.verify_jwt_token),
        ):
            return {"token": payload}

    Configuration:
        - JWT_SECRET_KEY: Required secret key for token verification
        - JWT_ALGORITHM: Algorithm for JWT verification (default: HS256)
        - JWT_EXPIRY_SECONDS: Optional token expiry in seconds
        - APPROVED_USER_IDS: Comma-separated list of approved user IDs
        - ENABLE_ENCRYPTION: Toggle for user ID encryption (default: True)
        - ENCRYPTION_KEY: Hex-encoded key for AES-256-GCM encryption
        - AUTHORIZED_TOKEN_IPS: Comma-separated list of allowed IPs for /token endpoint
    """

    # HTTPBearer scheme used to extract the JWT from the Authorization header.
    _bearer_scheme = HTTPBearer()

    @classmethod
    def check_ip_authorization(cls, client_ip: str) -> HTTPAuthorizationCredentials:
        """Verify that the client IP is authorized to request tokens.

        This method checks if the provided client IP address is in the list
        of authorized IPs configured via the `AUTHORIZED_TOKEN_IPS` environment
        variable. The list always includes `127.0.0.1` and `::1` for local
        development.

        Args:
            client_ip: The IP address of the client making the request.

        Returns:
            HTTPAuthorizationCredentials: The credentials object (for FastAPI dependency injection).

        Raises:
            HTTPException 401: If the client IP is not in the authorized list.

        Example:
            @app.post("/token")
            async def get_token(
                credentials: HTTPAuthorizationCredentials = Depends(Auth.check_ip_authorization),
            ):
                # Token generation logic here
                pass
        """
        if client_ip not in config.authorized_token_ips:
            logger.warning(
                "Token request denied for IP '%s' (not in authorized list)", client_ip
            )
            raise HTTPException(
                status_code=401,
                detail=f"IP address '{client_ip}' is not authorized to request a token.",
            )

    @classmethod
    def check_jwt_is_configured(cls) -> str:
        """Verify that JWT authentication is properly configured.

        This method checks if the `JWT_SECRET_KEY` environment variable is set.
        The secret key is required for JWT token verification. If not configured,
        all authenticated requests will be rejected.

        Args:
            cls: The Auth class (used for classmethod definition).

        Returns:
            str: The configured JWT secret key.

        Raises:
            HTTPException 401: If JWT_SECRET_KEY is not set in the environment.

        Example:
            @app.post("/token")
            async def get_token(
                secret_key: str = Depends(Auth.check_jwt_is_configured),
            ):
                # Token generation logic here
                pass
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
        return secret_key

    @staticmethod
    def verify_jwt_token(
        credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
    ) -> dict:
        """Validate and decode a JWT Bearer token.

        This method extracts and decodes the JWT from the `Authorization: Bearer <token>`
        header. The token is verified against `JWT_SECRET_KEY` using the algorithm
        specified by `JWT_ALGORITHM` (default: `HS256`).

        **Expiry Behavior:**
        - When `JWT_EXPIRY_SECONDS` is **not** set, the `exp` claim is not enforced
          (tokens are accepted regardless of expiry).
        - When `JWT_EXPIRY_SECONDS` is set, expired tokens are rejected.

        Args:
            credentials: HTTPAuthorizationCredentials containing the Bearer token.
                Extracted automatically by FastAPI's HTTPBearer scheme.

        Returns:
            dict: The decoded JWT payload containing claims (e.g., `sub`, `iss`, `exp`).

        Raises:
            HTTPException 401: If:
                - The secret key is not configured (`JWT_SECRET_KEY` not set)
                - The token is malformed or has an invalid signature
                - The token is expired (when `JWT_EXPIRY_SECONDS` is configured)
                - The algorithm does not match `JWT_ALGORITHM`

        Example:
            @app.get("/protected")
            async def protected_route(
                payload: dict = Depends(Auth.verify_jwt_token),
            ):
                user_id = payload.get("sub")
                return {"user": user_id}
        """
        secret_key: str = Auth.check_jwt_is_configured()

        # When expiry is not configured we instruct python-jose to skip the `exp`
        # claim check entirely by passing options={"verify_exp": False}.
        decode_options = {}
        if config.jwt_expiry_seconds is None:
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

    @staticmethod
    def get_current_user(
        x_user_id: Annotated[Optional[str], Header()] = None,
        _token_payload: dict = Depends(verify_jwt_token),
    ) -> str:
        """Extract and validate the current user identifier from the request.

        This method performs a two-step validation process:
        1. JWT authentication via :func:`verify_jwt_token` (enforced first)
        2. User ID header validation against approved user list

        **Encryption Support:**
        When `ENABLE_ENCRYPTION` is enabled (default), the `X-User-ID` header
        value is decrypted using AES-256-GCM before validation. The `ENCRYPTION_KEY`
        environment variable must be set for this to work.

        Args:
            x_user_id: The value of the `X-User-ID` HTTP header.
                If not provided, the header must be present in the request.
            _token_payload: Decoded JWT payload from :func:`verify_jwt_token`.
                Included to enforce JWT validation before user ID checking.

        Returns:
            str: The validated and (if encrypted) decrypted user identifier.

        Raises:
            HTTPException 401: If:
                - The JWT is invalid or expired
                - The `X-User-ID` header is missing
                - The user ID is not in the `APPROVED_USER_IDS` list
            HTTPException 500: If encryption/decryption fails

        Example:
            @app.get("/user-info")
            async def user_info(
                current_user: str = Depends(Auth.get_current_user),
            ):
                return {"user": current_user}
        """
        user_id: str = x_user_id

        if config.enable_encryption:
            try:
                user_id = Encryption.decrypt(user_id)
            except Exception as err:
                raise HTTPException(
                    status_code=500,
                    detail=str(err)
                )

        try:
            Auth.validate_user(user_id)
        except Exception as err:
            raise HTTPException(
                status_code=401,
                detail=str(err)
            )

        return user_id

    @staticmethod
    def validate_user(user_id: str) -> None:
        """Validate that a user ID is in the approved list.

        This method checks if the provided user ID exists in the list of
        approved user identifiers configured via the `APPROVED_USER_IDS`
        environment variable.

        Args:
            user_id: The user identifier to validate.

        Returns:
            None: If the user ID is valid and approved.

        Raises:
            ValueError: If:
                - The user_id is empty or None
                - No approved users are configured (`APPROVED_USER_IDS` not set)
                - The user_id is not in the approved list

        Example:
            @app.get("/check-user")
            async def check_user(user_id: str = Header(...)):
                Auth.validate_user(user_id)
                return {"valid": True}
        """
        if not user_id:
            raise ValueError("Missing required: user-id",)

        approved = config.approved_user_ids
        if not approved:
            raise ValueError(
                (
                    "No approved users configured. "
                    "Set APPROVED_USER_IDS in the environment."
                )
            )

        if user_id not in approved:
            logger.warning(
                "Invalidated & unknown user '%s'",
                user_id
            )
            raise ValueError(
                f"User ID '{user_id}' is not authorized.",
            )
