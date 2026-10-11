from __future__ import annotations

import re
from typing import Any


_SENSITIVE_KEY = re.compile(
    r"(?:authorization|api[_-]?key|access[_-]?token|secret|password|cookie)", re.I
)
_BEARER = re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+")
_OPENAI_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")


def redact_text(value: str, secrets: tuple[str, ...] = ()) -> str:
    value = _BEARER.sub("Bearer [REDACTED]", value)
    value = _OPENAI_KEY.sub("[REDACTED]", value)
    for secret in secrets:
        if secret:
            value = value.replace(secret, "[REDACTED]")
    return value


def redact(value: Any, secrets: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if _SENSITIVE_KEY.search(str(key)) else redact(item, secrets)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item, secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item, secrets) for item in value)
    if isinstance(value, str):
        return redact_text(value, secrets)
    return value


def safe_component(value: str) -> str:
    """Return a filesystem-safe, stable component without leaking raw identifiers."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError("identifier has no safe filesystem representation")
    return cleaned[:160]
