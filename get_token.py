from __future__ import annotations

import os
import base64
from binascii import Error as BinasciiError
from typing import Optional

from dotenv import load_dotenv
from gigachat import GigaChat


# Load variables from a local .env file if present
load_dotenv()

# Supported environment variable names for the API key
ENV_KEYS = ("GIGACHAT_API_KEY", "GIGACHAT_CREDENTIALS", "GIGACHAT_AUTHORIZATION")


def _normalize_credentials(raw_value: str) -> str:
    """Return base64 credentials without the 'Basic ' prefix if present.

    Accepts either:
    - plain base64 string
    - full header value like 'Basic abcd...'
    """
    # Trim quotes and spaces/newlines
    value = raw_value.strip().strip('"').strip("'")
    lower = value.lower()
    # Allow full header copy-paste like 'Authorization: Basic xxx'
    if lower.startswith("authorization:"):
        value = value.split(":", 1)[1].strip()
        lower = value.lower()
    # Remove 'Basic ' prefix if present
    if lower.startswith("basic "):
        value = value[6:].strip()
    # Remove all inner whitespace/newlines that may appear from wrapped copies
    value = "".join(value.split())
    return value


def _ensure_base64_credentials(value: str) -> str:
    """Validate the credentials look like base64; return normalized string.

    If not valid base64, raise ValueError with guidance.
    """
    normalized = _normalize_credentials(value)
    def _pad(s: str) -> str:
        return s + ("=" * (-len(s) % 4))

    def _to_b64(s: str) -> str:
        # Convert base64url to base64 if needed
        if "-" in s or "_" in s:
            s = s.replace("-", "+").replace("_", "/")
        return s

    # Try as-is
    try:
        base64.b64decode(normalized, validate=True)
        return normalized
    except Exception:
        pass

    # Try with padding
    padded = _pad(normalized)
    try:
        base64.b64decode(padded, validate=True)
        return padded
    except Exception:
        pass

    # Try base64url conversion
    converted = _to_b64(normalized)
    try:
        base64.b64decode(converted, validate=True)
        return converted
    except Exception:
        pass

    # Try base64url + padding
    converted_padded = _pad(converted)
    try:
        base64.b64decode(converted_padded, validate=True)
        return converted_padded
    except Exception:
        raise ValueError(
            "Invalid credentials format. Provide base64 Authorization data (without 'Basic '), "
            "or set GIGACHAT_CLIENT_ID and GIGACHAT_CLIENT_SECRET to auto-generate it."
        )


def get_gigachat_token(
    credentials: Optional[str] = None,
    *,
    verify_ssl_certs: Optional[bool] = None,
    ca_bundle_file: Optional[str] = None,
) -> str:
    """Return a GigaChat API token.

    Looks up the API key in this order:
    1) The provided ``credentials`` argument
    2) Environment variables: ``GIGACHAT_API_KEY`` or ``GIGACHAT_CREDENTIALS`` (from OS or .env)

    Raises:
        ValueError: If no API key is provided or found in the environment.
    """

    # 1) Provided explicitly
    api_key = credentials

    # 2) Env variables including possible 'Authorization: Basic ...'
    if not api_key:
        api_key = next((os.getenv(k) for k in ENV_KEYS if os.getenv(k)), None)

    # 3) Build from client id/secret when available
    if not api_key:
        client_id = os.getenv("GIGACHAT_CLIENT_ID")
        client_secret = os.getenv("GIGACHAT_CLIENT_SECRET")
        if client_id and client_secret:
            pair = f"{client_id}:{client_secret}".encode("utf-8")
            api_key = base64.b64encode(pair).decode("ascii")

    if not api_key:
        raise ValueError(
            "GigaChat API key not provided. Set one of "
            f"{', '.join(ENV_KEYS)} or GIGACHAT_CLIENT_ID/GIGACHAT_CLIENT_SECRET in your environment, "
            "or pass 'credentials' argument."
        )

    # Normalize and validate base64 value
    api_key = _ensure_base64_credentials(api_key)

    # SSL options from args or env
    if verify_ssl_certs is None:
        env_verify = os.getenv("GIGACHAT_VERIFY_SSL")
        if env_verify is not None:
            verify_ssl_certs = env_verify.strip().lower() in {"1", "true", "yes", "on"}
    if ca_bundle_file is None:
        ca_bundle_file = os.getenv("GIGACHAT_CA_BUNDLE_FILE")
        # Fallback to local trusted root if present
        if not ca_bundle_file:
            local_ca = os.path.join(os.path.dirname(__file__), "russian_trusted_root_ca.cer")
            if os.path.exists(local_ca):
                ca_bundle_file = local_ca

    giga = GigaChat(
        credentials=api_key,
        verify_ssl_certs=verify_ssl_certs,
        ca_bundle_file=ca_bundle_file,
    )
    response = giga.get_token()
    return response


if __name__ == "__main__":
    # Allows quick manual run: `python get_token.py`
    print(get_gigachat_token())