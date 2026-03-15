"""Message transforms (e.g., middle-out truncation)."""

from __future__ import annotations

from typing import Any


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return len(text) // 4


def _message_tokens(message: dict[str, Any]) -> int:
    """Estimate tokens in a single message."""
    content = message.get("content", "")
    if isinstance(content, str):
        return _estimate_tokens(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                total += _estimate_tokens(part.get("text", ""))
        return total
    return 0


def middle_out(messages: list[dict[str, Any]], context_length: int) -> list[dict[str, Any]]:
    """Remove messages from the middle to fit within context_length tokens.

    Preserves the system prompt (first message) and most recent messages.
    """
    total = sum(_message_tokens(m) for m in messages)
    if total <= context_length:
        return messages

    # Always keep first message (system) and last message
    if len(messages) <= 2:
        return messages

    result = list(messages)
    # Remove from the middle until we fit
    while len(result) > 2:
        current = sum(_message_tokens(m) for m in result)
        if current <= context_length:
            break
        # Remove the message just after the system prompt
        mid = len(result) // 2
        result.pop(mid)

    return result


def apply_transforms(
    transforms: list[str],
    messages: list[dict[str, Any]],
    context_length: int = 128000,
) -> list[dict[str, Any]]:
    """Apply named transforms to a message list."""
    result = messages
    for name in transforms:
        if name == "middle-out":
            result = middle_out(result, context_length)
    return result
