"""Verify Firebase ID tokens from the website."""
from __future__ import annotations

import logging
import os

from fastapi import HTTPException

PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "sovereign-76855")
JWKS_URL = "https://www.googleapis.com/service_accounts/v1/jwk/securetoken@system.gserviceaccount.com"
_logger = logging.getLogger("sovereign.auth")
_jwk_client = None


def _debug_auth(message: str, *args: object) -> None:
    if os.environ.get("SOVEREIGN_AUTH_DEBUG") == "1":
        _logger.warning(message, *args)


def verify_firebase_token(token: str) -> dict:
    if os.environ.get("SOVEREIGN_TEST_AUTH") == "1" and token == "test-firebase-token":
        return {"sub": "test-user-id", "email": "test@example.com"}
    try:
        return _verify_with_jwks(token)
    except Exception as exc:
        _debug_auth("JWKS verify failed: %s", exc)
        try:
            from google.auth.transport import requests as google_requests
            from google.oauth2 import id_token
            return id_token.verify_firebase_token(
                token, google_requests.Request(), audience=PROJECT_ID)
        except Exception as fallback_exc:
            _debug_auth("google-auth verify failed: %s", fallback_exc)
            raise HTTPException(401, "Sign in to continue.") from fallback_exc


def _verify_with_jwks(token: str) -> dict:
    global _jwk_client
    import jwt
    from jwt import PyJWKClient

    if _jwk_client is None:
        _jwk_client = PyJWKClient(JWKS_URL, cache_keys=True)
    signing_key = _jwk_client.get_signing_key_from_jwt(token)
    return jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        audience=PROJECT_ID,
        issuer=f"https://securetoken.google.com/{PROJECT_ID}",
    )
