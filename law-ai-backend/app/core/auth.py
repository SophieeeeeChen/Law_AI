import time
from typing import Optional

import jwt
import requests
from fastapi import Header, HTTPException, status

from app.core.config import Config


_JWKS_CACHE = {"keys": None, "fetched_at": 0}
_JWKS_TTL_SECONDS = 3600


def _get_jwks_url() -> Optional[str]:
    if Config.ENTRA_JWKS_URL:
        return Config.ENTRA_JWKS_URL
    if Config.ENTRA_TENANT_ID:
        return f"https://login.microsoftonline.com/{Config.ENTRA_TENANT_ID}/discovery/v2.0/keys"
    return None


def _fetch_jwks():
    jwks_url = _get_jwks_url()
    if not jwks_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="ENTRA_JWKS_URL or ENTRA_TENANT_ID must be set for Entra auth.",
        )
    now = time.time()
    if _JWKS_CACHE["keys"] and (now - _JWKS_CACHE["fetched_at"] < _JWKS_TTL_SECONDS):
        return _JWKS_CACHE["keys"]
    resp = requests.get(jwks_url, timeout=5)
    resp.raise_for_status()
    _JWKS_CACHE["keys"] = resp.json().get("keys", [])
    _JWKS_CACHE["fetched_at"] = now
    return _JWKS_CACHE["keys"]


def _get_rsa_key(token: str):
    unverified_header = jwt.get_unverified_header(token)
    kid = unverified_header.get("kid")
    for key in _fetch_jwks():
        if key.get("kid") == kid:
            return jwt.algorithms.RSAAlgorithm.from_jwk(key)
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token key id.")


def _validate_entra_jwt(token: str) -> dict:
    rsa_key = _get_rsa_key(token)
    audience = Config.ENTRA_AUDIENCE or Config.ENTRA_CLIENT_ID
    issuer = Config.ENTRA_ISSUER
    options = {
        "verify_aud": bool(audience),
        "verify_iss": bool(issuer),
    }
    return jwt.decode(
        token,
        rsa_key,
        algorithms=["RS256"],
        audience=audience,
        issuer=issuer,
        options=options,
    )


def get_current_user_id(
    authorization: Optional[str] = Header(None),
    x_dev_user: Optional[str] = Header(None),
) -> Optional[str]:
    auth_mode = Config.AUTH_MODE
    if not auth_mode:
        auth_mode = "entra" if Config.ENV == "prd" else "dev"
    if auth_mode == "dev":
        return x_dev_user

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token.")

    token = authorization.split(" ", 1)[1]
    try:
        claims = _validate_entra_jwt(token)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token.") from exc

    user_id = claims.get("oid") or claims.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing user id.")
    return user_id
