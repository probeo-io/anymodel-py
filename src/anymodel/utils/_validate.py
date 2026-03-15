"""Request validation."""

from __future__ import annotations

from typing import Any

from anymodel._types import AnyModelError


def validate_request(request: dict[str, Any]) -> None:
    """Validate a chat completion request dict."""
    if not request.get("model") and not request.get("models"):
        raise AnyModelError(400, "Either 'model' or 'models' is required.")

    messages = request.get("messages")
    if not messages or not isinstance(messages, list):
        raise AnyModelError(400, "'messages' must be a non-empty list.")

    temp = request.get("temperature")
    if temp is not None and not (0 <= temp <= 2):
        raise AnyModelError(400, "'temperature' must be between 0 and 2.")

    top_p = request.get("top_p")
    if top_p is not None and not (0 <= top_p <= 1):
        raise AnyModelError(400, "'top_p' must be between 0 and 1.")

    stop = request.get("stop")
    if isinstance(stop, list) and len(stop) > 4:
        raise AnyModelError(400, "'stop' must have at most 4 entries.")
