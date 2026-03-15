"""Provider registry."""

from __future__ import annotations

from anymodel._types import AnyModelError
from anymodel.providers._adapter import ProviderAdapter


class ProviderRegistry:
    """Registry of configured provider adapters."""

    def __init__(self) -> None:
        self._adapters: dict[str, ProviderAdapter] = {}

    def register(self, slug: str, adapter: ProviderAdapter) -> None:
        """Register a provider adapter."""
        if slug in self._adapters:
            raise AnyModelError(500, f"Provider '{slug}' is already registered.")
        self._adapters[slug] = adapter

    def get(self, slug: str) -> ProviderAdapter:
        """Get a provider adapter by slug."""
        adapter = self._adapters.get(slug)
        if adapter is None:
            raise AnyModelError(400, f"Provider '{slug}' not configured.")
        return adapter

    def has(self, slug: str) -> bool:
        """Check if a provider is registered."""
        return slug in self._adapters

    def all(self) -> list[ProviderAdapter]:
        """Return all registered adapters."""
        return list(self._adapters.values())

    def slugs(self) -> list[str]:
        """Return all registered provider slugs."""
        return list(self._adapters.keys())
