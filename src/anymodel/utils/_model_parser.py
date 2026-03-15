"""Model string parsing utilities."""

from __future__ import annotations

from typing import NamedTuple

from anymodel._types import AnyModelError


class ParsedModel(NamedTuple):
    provider: str
    model: str


def parse_model_string(
    model: str,
    aliases: dict[str, str] | None = None,
) -> ParsedModel:
    """Parse a 'provider/model' string, resolving aliases first."""
    if aliases and model in aliases:
        model = aliases[model]

    if "/" not in model:
        raise AnyModelError(400, f"Invalid model format '{model}'. Expected 'provider/model'.")

    provider, _, model_name = model.partition("/")
    if not provider or not model_name:
        raise AnyModelError(400, f"Invalid model format '{model}'. Expected 'provider/model'.")

    return ParsedModel(provider=provider, model=model_name)
