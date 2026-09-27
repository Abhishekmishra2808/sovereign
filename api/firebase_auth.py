"""Verify Firebase ID tokens from the website."""
from __future__ import annotations

import os

from fastapi import HTTPException

PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "sovereign-76855")


def verify_firebase_token(token: str) -> dict:
    if os.environ.get("SOVEREIGN_TEST_AUTH") == "1" and token == "test-firebase-token":
        return {"sub": "test-user-id", "email": "test@example.com"}
    try:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token
        return id_token.verify_firebase_token(
            token, google_requests.Request(), audience=PROJECT_ID)
    except Exception as exc:
        raise HTTPException(401, "Sign in to continue.") from exc
