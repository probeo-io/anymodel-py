"""Prompt cache key helpers — provider-neutral caching support."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any


def _stable_json(value: Any) -> str:
    """Serialize *value* with sorted keys for a deterministic hash input."""
    if value is None or not isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ",".join(_stable_json(v) for v in value) + "]"
    entries = sorted((k, v) for k, v in value.items() if v is not None)
    return "{" + ",".join(f"{json.dumps(k)}:{_stable_json(v)}" for k, v in entries) + "}"


def create_prompt_cache_key(parts: Any, *, prefix: str | None = None, max_length: int = 128) -> str:
    """Create a stable prompt cache key from inputs that define a reusable prompt
    prefix, such as workflow version, domain, schema version, and archetype set.
    """
    digest = hashlib.sha256(_stable_json(parts).encode()).digest()
    encoded = base64.urlsafe_b64encode(digest).decode().rstrip("=")
    key = f"{prefix}:{encoded}" if prefix else encoded
    return key[:max_length]


def with_prompt_cache(request: dict[str, Any], cache: dict[str, Any]) -> dict[str, Any]:
    """Attach provider-neutral prompt cache options to a chat completion request."""
    return {**request, "cache": {**request.get("cache", {}), **cache}}
