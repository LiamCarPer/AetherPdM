"""API key authentication for the AetherPdM API.

Keys are sent via the X-API-Key header. The dependency verifies the key
against the ApiKey table (hashed). When AETHER_API_KEY_AUTH_ENABLED is
false (default, dev), the dependency is a no-op and all requests pass.
When true, /v1/* routes require a valid key; /health, /metrics, /docs,
/openapi.json remain open.

Key format: aether_<prefix>_<secret>  (e.g. aether_Ab12cD34_<random>)
"""

import hashlib
import hmac
import os
import secrets
from collections.abc import Generator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from aether_pdm.db.database import get_session
from aether_pdm.db.repository import get_api_key_by_prefix

KEY_PREFIX_LEN = 8
KEY_SECRET_LEN = 24  # raw random bytes -> 48-char hex secret

DEFAULT_ORG = "default"


class Tenant(BaseModel):
    """Authenticated tenant context (org + key name)."""

    org: str
    key_name: str | None

    model_config = {"frozen": True}


def default_tenant() -> Tenant:
    """Tenant used when auth is disabled (dev mode)."""
    return Tenant(org=DEFAULT_ORG, key_name=None)


def auth_enabled() -> bool:
    """Whether API key auth is enabled (env AETHER_API_KEY_AUTH_ENABLED)."""
    return os.getenv("AETHER_API_KEY_AUTH_ENABLED", "false").lower() in {"1", "true", "yes"}


def hash_api_key(secret_part: str) -> str:
    """PBKDF2-HMAC-SHA256 hash of the secret part (with per-key salt)."""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", secret_part.encode(), salt, 100_000)
    return salt.hex() + ":" + dk.hex()


def verify_api_key_hash(secret_part: str, stored_hash: str) -> bool:
    """Constant-time comparison of a candidate secret against the stored hash."""
    try:
        salt_hex, dk_hex = stored_hash.split(":")
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(dk_hex)
    except ValueError:
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", secret_part.encode(), salt, 100_000)
    return hmac.compare_digest(candidate, expected)


def generate_api_key() -> tuple[str, str, str]:
    """
    Generate a new API key.

    Returns (full_key, prefix, secret_part):
      full_key = "aether_" + prefix + "_" + secret  (shown ONCE to the user)
      prefix   = first KEY_PREFIX_LEN chars of the random secret
      secret_part = the random secret (hashed for storage)

    Uses hex (not base64url) so the secret never contains "_": the
    aether_<prefix>_<secret> format stays unambiguous for parsing.
    """
    raw = secrets.token_hex(KEY_SECRET_LEN)
    prefix = raw[:KEY_PREFIX_LEN]
    secret_part = raw
    full_key = f"aether_{prefix}_{secret_part}"
    return full_key, prefix, secret_part


def _resolve_key(x_api_key: str) -> tuple[str, str] | None:
    """
    Given the full header value, split into (prefix, secret_part).

    Accepts either "aether_<prefix>_<secret>" or a bare "<prefix><secret>"
    (robustness). Returns None if malformed.
    """
    if x_api_key.startswith("aether_"):
        # format: aether_<prefix>_<secret>
        parts = x_api_key.split("_", 2)
        if len(parts) != 3:
            return None
        return parts[1], parts[2]
    # bare format: prefix + secret (prefix is first KEY_PREFIX_LEN chars)
    return x_api_key[:KEY_PREFIX_LEN], x_api_key[KEY_PREFIX_LEN:]


def _get_db() -> Generator[Session, None, None]:
    """Session dependency mirroring ``serve.app.get_db`` (avoids circular import)."""
    with get_session() as session:
        yield session


async def require_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    db: Session = Depends(_get_db),
) -> Tenant:
    """
    FastAPI dependency: validate the X-API-Key header.

    - If auth disabled: return ``default_tenant()`` (org "default", dev mode).
    - If auth enabled and header present: verify prefix + hash and return the
      tenant context derived from the API key record (org + key name).
    - If auth enabled and header missing/invalid: raise 401.

    Async so DB lookups run on the event loop (matching the async endpoints);
    a plain sync dependency would execute in a worker thread, which breaks
    in-memory SQLite test setups that rely on per-thread connections.

    Returns the tenant (org + key name) for downstream scoping / logging.
    """
    if not auth_enabled():
        return default_tenant()

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )

    resolved = _resolve_key(x_api_key)
    if resolved is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format",
        )

    prefix, secret = resolved
    record = get_api_key_by_prefix(db, prefix)
    if record is None or record.revoked_at is not None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    if not verify_api_key_hash(secret, record.key_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    return Tenant(org=record.org, key_name=record.name)
