"""Provider-specific pricing policy for batch/service-tier discounts."""

from __future__ import annotations

from typing import Literal

PricingMode = Literal["standard", "flex", "native_batch"]

_POLICY: dict[str, dict[str, float]] = {
    "openai": {"standard": 1.0, "flex": 0.5, "native_batch": 0.5},
    "anthropic": {"standard": 1.0, "native_batch": 0.5},
    "google": {"standard": 1.0, "native_batch": 0.5},
    "xai": {"standard": 1.0},
    "perplexity": {"standard": 1.0},
}


def provider_pricing_multiplier(provider: str, mode: PricingMode) -> float:
    """Return the cost multiplier for a provider/mode combination."""
    return _POLICY.get(provider, {}).get(mode, 1.0)
