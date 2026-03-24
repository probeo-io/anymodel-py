"""AnyModel client — main entry point."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from typing import Any, Literal, overload

from anymodel._config import resolve_config
from anymodel._router import Router
from anymodel.batch._builder import BatchBuilder
from anymodel.batch._manager import BatchManager
from anymodel.providers._anthropic import create_anthropic_adapter
from anymodel.providers._anthropic_batch import create_anthropic_batch_adapter
from anymodel.providers._custom import create_custom_adapter
from anymodel.providers._google import create_google_adapter
from anymodel.providers._google_batch import create_google_batch_adapter
from anymodel.providers._openai import create_openai_adapter
from anymodel.providers._openai_batch import create_openai_batch_adapter
from anymodel.providers._perplexity import create_perplexity_adapter
from anymodel.providers._registry import ProviderRegistry
from anymodel.utils._fs_io import configure_fs_io
from anymodel.utils._generation_stats import GenerationStatsStore
from anymodel.utils._timeout import set_default_timeout

# Built-in OpenAI-compatible providers
_BUILTIN_PROVIDERS = [
    {"name": "mistral", "base_url": "https://api.mistral.ai/v1", "env_var": "MISTRAL_API_KEY"},
    {"name": "groq", "base_url": "https://api.groq.com/openai/v1", "env_var": "GROQ_API_KEY"},
    {"name": "deepseek", "base_url": "https://api.deepseek.com/v1", "env_var": "DEEPSEEK_API_KEY"},
    {"name": "xai", "base_url": "https://api.x.ai/v1", "env_var": "XAI_API_KEY"},
    {"name": "together", "base_url": "https://api.together.xyz/v1", "env_var": "TOGETHER_API_KEY"},
    {"name": "fireworks", "base_url": "https://api.fireworks.ai/inference/v1", "env_var": "FIREWORKS_API_KEY"},
]


class _ChatCompletions:
    """Namespace for chat.completions.create()."""

    def __init__(self, client: AnyModel) -> None:
        self._client = client

    @overload
    async def create(self, *, stream: Literal[True], **kwargs: Any) -> AsyncIterator[dict[str, Any]]: ...

    @overload
    async def create(self, *, stream: Literal[False] = ..., **kwargs: Any) -> dict[str, Any]: ...

    @overload
    async def create(self, **kwargs: Any) -> dict[str, Any] | AsyncIterator[dict[str, Any]]: ...

    async def create(self, **kwargs: Any) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """Create a chat completion."""
        if kwargs.get("stream"):
            return await self._client._router.stream(kwargs)
        return await self._client._router.complete(kwargs)


class _Chat:
    """Namespace for chat.completions."""

    def __init__(self, client: AnyModel) -> None:
        self.completions = _ChatCompletions(client)


class _Models:
    """Namespace for models.list()."""

    def __init__(self, client: AnyModel) -> None:
        self._client = client

    async def list(self, *, provider: str | None = None) -> list[dict[str, Any]]:
        """List available models, optionally filtered by provider."""
        all_models: list[dict[str, Any]] = []
        adapters = self._client._registry.all()
        for adapter in adapters:
            if provider and adapter.name != provider:
                continue
            try:
                models = await adapter.list_models()
                all_models.extend(models)
            except Exception:
                continue
        return all_models


class _Generation:
    """Namespace for generation.get()."""

    def __init__(self, client: AnyModel) -> None:
        self._client = client

    def get(self, gen_id: str) -> dict[str, Any] | None:
        """Get generation stats by ID."""
        return self._client._stats_store.get(gen_id)

    def list(self) -> list[dict[str, Any]]:
        """List all generation stats."""
        return self._client._stats_store.list()


class _Batches:
    """Namespace for batch operations."""

    def __init__(self, client: AnyModel) -> None:
        self._client = client

    def open(self, config: dict[str, Any]) -> BatchBuilder:
        """Open a new batch builder for incremental prompt addition."""
        return BatchBuilder(config, self._client._batch_manager.get_store(), self._client._batch_manager)

    async def create(self, request: dict[str, Any]) -> dict[str, Any]:
        """Create a batch (fire-and-forget for native, background for concurrent)."""
        return await self._client._batch_manager.create(request)

    async def create_and_poll(
        self,
        request: dict[str, Any],
        *,
        interval: float | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Create a batch and poll until completion."""
        return await self._client._batch_manager.create_and_poll(
            request, interval=interval, on_progress=on_progress,
        )

    async def poll(
        self,
        batch_id: str,
        *,
        interval: float | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Poll a batch until completion."""
        return await self._client._batch_manager.poll(
            batch_id, interval=interval, on_progress=on_progress,
        )

    async def get(self, batch_id: str) -> dict[str, Any] | None:
        """Get batch status."""
        return await self._client._batch_manager.get(batch_id)

    async def list(self) -> list[dict[str, Any]]:
        """List all batches."""
        return await self._client._batch_manager.list()

    async def results(self, batch_id: str) -> dict[str, Any]:
        """Get batch results."""
        return await self._client._batch_manager.results(batch_id)

    async def cancel(self, batch_id: str) -> dict[str, Any]:
        """Cancel a batch."""
        return await self._client._batch_manager.cancel(batch_id)


class AnyModel:
    """Unified LLM client that routes across multiple providers."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = resolve_config(config)
        self._registry = ProviderRegistry()
        self._stats_store = GenerationStatsStore()

        # Configure default HTTP timeout
        timeout = self._config.get("defaults", {}).get("timeout", 120.0)
        set_default_timeout(timeout)

        # Configure IO concurrency
        io_config = self._config.get("io")
        if io_config:
            configure_fs_io(
                read_concurrency=io_config.get("read_concurrency"),
                write_concurrency=io_config.get("write_concurrency"),
            )

        self._router = Router(
            self._registry,
            aliases=self._config.get("aliases"),
            defaults=self._config.get("defaults"),
            stats_store=self._stats_store,
        )

        batch_config = self._config.get("batch", {})
        self._batch_manager = BatchManager(
            self._router,
            dir=batch_config.get("dir"),
            concurrency=batch_config.get("concurrency_fallback", 5),
            poll_interval=batch_config.get("poll_interval", 5.0),
            aliases=self._config.get("aliases"),
        )

        self._register_providers()
        self._register_batch_adapters()

        # Namespace objects
        self.chat = _Chat(self)
        self.models = _Models(self)
        self.generation = _Generation(self)
        self.batches = _Batches(self)

    def _register_providers(self) -> None:
        """Register all configured providers."""
        # OpenAI
        openai_key = self._config.get("openai", {}).get("api_key") or os.environ.get("OPENAI_API_KEY")
        if openai_key:
            self._registry.register("openai", create_openai_adapter(openai_key))

        # Anthropic
        anthropic_key = self._config.get("anthropic", {}).get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            self._registry.register("anthropic", create_anthropic_adapter(anthropic_key))

        # Google
        google_key = self._config.get("google", {}).get("api_key") or os.environ.get("GOOGLE_API_KEY")
        if google_key:
            self._registry.register("google", create_google_adapter(google_key))

        # Perplexity
        perplexity_key = self._config.get("perplexity", {}).get("api_key") or os.environ.get("PERPLEXITY_API_KEY")
        if perplexity_key:
            self._registry.register("perplexity", create_perplexity_adapter(perplexity_key))

        # Built-in OpenAI-compatible providers
        for p in _BUILTIN_PROVIDERS:
            provider_config = self._config.get(p["name"], {})
            key = provider_config.get("api_key") or os.environ.get(p["env_var"])
            if key:
                self._registry.register(
                    p["name"],
                    create_custom_adapter(p["name"], p["base_url"], key),
                )

        # Ollama
        ollama_config = self._config.get("ollama", {})
        ollama_url = ollama_config.get("base_url") or os.environ.get("OLLAMA_BASE_URL")
        if ollama_url:
            self._registry.register(
                "ollama",
                create_custom_adapter("ollama", ollama_url),
            )

        # Custom providers
        custom = self._config.get("custom", {})
        for name, cfg in custom.items():
            self._registry.register(
                name,
                create_custom_adapter(
                    name,
                    cfg["base_url"],
                    cfg.get("api_key", ""),
                    cfg.get("models"),
                ),
            )

    def _register_batch_adapters(self) -> None:
        """Register native batch adapters for providers that support them."""
        openai_key = self._config.get("openai", {}).get("api_key") or os.environ.get("OPENAI_API_KEY")
        if openai_key:
            self._batch_manager.register_batch_adapter("openai", create_openai_batch_adapter(openai_key))

        anthropic_key = self._config.get("anthropic", {}).get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if anthropic_key:
            self._batch_manager.register_batch_adapter("anthropic", create_anthropic_batch_adapter(anthropic_key))

        google_key = self._config.get("google", {}).get("api_key") or os.environ.get("GOOGLE_API_KEY")
        if google_key:
            self._batch_manager.register_batch_adapter("google", create_google_batch_adapter(google_key))
