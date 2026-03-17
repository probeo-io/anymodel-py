"""Token estimation utilities for automatic max_tokens resolution."""

from __future__ import annotations

import json
from typing import Any

# Approximate characters per token
_CHARS_PER_TOKEN = 4

# Model limits: model_prefix -> (context_length, max_completion_tokens)
_MODEL_LIMITS: dict[str, tuple[int, int]] = {
    # OpenAI
    "gpt-4o-mini": (128_000, 16_384),
    "gpt-4o": (128_000, 16_384),
    "gpt-4-turbo": (128_000, 4_096),
    "gpt-3.5-turbo": (16_385, 4_096),
    "o1": (200_000, 100_000),
    "o3": (200_000, 100_000),
    "o4-mini": (200_000, 100_000),
    # Anthropic
    "claude-opus-4": (200_000, 32_768),
    "claude-sonnet-4": (200_000, 16_384),
    "claude-haiku-4": (200_000, 8_192),
    "claude-3.5-sonnet": (200_000, 8_192),
    "claude-3-opus": (200_000, 4_096),
    # Google
    "gemini-2.5-pro": (1_048_576, 65_536),
    "gemini-2.5-flash": (1_048_576, 65_536),
    "gemini-2.0-flash": (1_048_576, 65_536),
    "gemini-1.5-pro": (2_097_152, 8_192),
    "gemini-1.5-flash": (1_048_576, 8_192),
}

_DEFAULT_LIMITS = (128_000, 4_096)


def estimate_token_count(text: str) -> int:
    """Estimate token count from character count (~4 chars per token)."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def get_model_limits(model: str) -> tuple[int, int]:
    """Look up context length and max completion tokens for a model.

    Strips any ``provider/`` prefix and does longest-prefix matching
    against known models.  Returns ``(context_length, max_completion_tokens)``.
    """
    # Strip provider prefix (e.g. "openai/gpt-4o" -> "gpt-4o")
    if "/" in model:
        model = model.split("/", 1)[1]

    # Try exact match first, then prefix match (longest prefix wins)
    if model in _MODEL_LIMITS:
        return _MODEL_LIMITS[model]

    best: tuple[int, int] | None = None
    best_len = 0
    for prefix, limits in _MODEL_LIMITS.items():
        if model.startswith(prefix) and len(prefix) > best_len:
            best = limits
            best_len = len(prefix)

    return best if best is not None else _DEFAULT_LIMITS


def resolve_max_tokens(
    model: str,
    messages: list[dict[str, Any]],
    user_max_tokens: int | None = None,
) -> int:
    """Resolve the max_tokens value for a request.

    If *user_max_tokens* is provided, it is returned as-is (the user
    knows what they want).  Otherwise the value is estimated from the
    message payload size, model context window, and model completion
    limit.
    """
    if user_max_tokens is not None:
        return user_max_tokens

    # Estimate input tokens from the JSON-serialised messages
    input_chars = len(json.dumps(messages, default=str))
    estimated_input = input_chars // _CHARS_PER_TOKEN

    # Add 5 % safety margin
    estimated_with_margin = int(estimated_input * 1.05)

    context_length, max_completion = get_model_limits(model)

    remaining = context_length - estimated_with_margin
    return max(1, min(max_completion, remaining))
