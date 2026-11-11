"""Auth settings from the environment. Secret values are never logged or put in errors."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cryptography.fernet import Fernet

SECRET_VARS = (
    "YAHOO_CLIENT_ID",
    "YAHOO_CLIENT_SECRET",
    "JWT_SIGNING_KEY",
    "STORAGE_ENCRYPTION_KEY",
)
REQUIRED_VARS = (*SECRET_VARS, "STATLINE_BASE_URL")


@dataclass(frozen=True)
class AuthConfig:
    yahoo_client_id: str
    yahoo_client_secret: str
    jwt_signing_key: str
    storage_encryption_key: bytes
    base_url: str
    token_store: Literal["firestore", "disk"] = "firestore"
    token_dir: Path = Path("data/oauth")
    google_cloud_project: str | None = None


def load_auth_config(env: Mapping[str, str] = os.environ) -> AuthConfig | None:
    """Return None when no auth variable is set (auth disabled); raise on a partial setup."""
    if not any(env.get(name) for name in SECRET_VARS):
        return None
    missing = [name for name in REQUIRED_VARS if not env.get(name)]
    store = env.get("STATLINE_TOKEN_STORE") or "firestore"
    if store not in ("firestore", "disk"):
        raise ValueError(f"STATLINE_TOKEN_STORE must be 'firestore' or 'disk', got {store!r}")
    if store == "firestore" and not env.get("GOOGLE_CLOUD_PROJECT"):
        missing.append("GOOGLE_CLOUD_PROJECT")
    if missing:
        raise ValueError(f"auth is partially configured; missing env: {', '.join(missing)}")
    key = env["STORAGE_ENCRYPTION_KEY"].encode()
    try:
        Fernet(key)
    except ValueError as exc:
        raise ValueError("STORAGE_ENCRYPTION_KEY is not a valid Fernet key") from exc
    return AuthConfig(
        yahoo_client_id=env["YAHOO_CLIENT_ID"],
        yahoo_client_secret=env["YAHOO_CLIENT_SECRET"],
        jwt_signing_key=env["JWT_SIGNING_KEY"],
        storage_encryption_key=key,
        base_url=env["STATLINE_BASE_URL"].rstrip("/"),
        token_store=store,  # type: ignore[arg-type]
        token_dir=Path(env.get("STATLINE_TOKEN_DIR") or "data/oauth"),
        google_cloud_project=env.get("GOOGLE_CLOUD_PROJECT") or None,
    )
