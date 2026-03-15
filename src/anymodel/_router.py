"""Request routing with fallback support."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

from anymodel._types import AnyModelError
from anymodel.providers._registry import ProviderRegistry
from anymodel.utils._generation_stats import GenerationStatsStore
from anymodel.utils._id import generate_id
from anymodel.utils._model_parser import parse_model_string
from anymodel.utils._rate_limiter import RateLimitTracker
from anymodel.utils._retry import with_retry
from anymodel.utils._transforms import apply_transforms
from anymodel.utils._validate import validate_request


class Router:
    """Routes chat completion requests to the appropriate provider."""

    def __init__(
        self,
        registry: ProviderRegistry,
        *,
        aliases: dict[str, str] | None = None,
        defaults: dict[str, Any] | None = None,
        stats_store: GenerationStatsStore | None = None,
    ) -> None:
        self._registry = registry
        self._aliases = aliases or {}
        self._defaults = defaults or {}
        self._rate_limiter = RateLimitTracker()
        self._stats_store = stats_store or GenerationStatsStore()

    @property
    def registry(self) -> ProviderRegistry:
        return self._registry

    async def complete(self, request: dict[str, Any]) -> dict[str, Any]:
        """Route a chat completion request."""
        request = self._apply_defaults(request)
        validate_request(request)

        # Apply transforms
        transforms = request.pop("transforms", None)
        if transforms:
            request["messages"] = apply_transforms(transforms, request["messages"])

        # Fallback routing
        models = request.get("models")
        if models and request.get("route") == "fallback":
            return await self._complete_with_fallback(request, models)

        # Single model
        parsed = parse_model_string(request["model"], self._aliases)
        adapter = self._registry.get(parsed.provider)

        # Strip unsupported params
        clean_request = self._strip_unsupported(request, adapter)
        clean_request["model"] = parsed.model

        retries = self._defaults.get("retries", 2)
        start = time.monotonic()

        result = await with_retry(
            lambda: adapter.send_request(clean_request),
            max_retries=retries,
        )

        elapsed = time.monotonic() - start
        self._record_stats(result, parsed.provider, elapsed, streamed=False)

        return result

    async def stream(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Route a streaming chat completion request."""
        request = self._apply_defaults(request)
        validate_request(request)

        transforms = request.pop("transforms", None)
        if transforms:
            request["messages"] = apply_transforms(transforms, request["messages"])

        # Fallback routing
        models = request.get("models")
        if models and request.get("route") == "fallback":
            return await self._stream_with_fallback(request, models)

        parsed = parse_model_string(request["model"], self._aliases)
        adapter = self._registry.get(parsed.provider)

        clean_request = self._strip_unsupported(request, adapter)
        clean_request["model"] = parsed.model

        retries = self._defaults.get("retries", 2)

        return await with_retry(
            lambda: adapter.send_streaming_request(clean_request),
            max_retries=retries,
        )

    async def _complete_with_fallback(
        self, request: dict[str, Any], models: list[str],
    ) -> dict[str, Any]:
        """Try models in order until one succeeds."""
        models = self._apply_provider_preferences(models, request.get("provider"))
        last_error: Exception | None = None

        for model in models:
            try:
                parsed = parse_model_string(model, self._aliases)

                if self._rate_limiter.is_rate_limited(parsed.provider):
                    continue

                adapter = self._registry.get(parsed.provider)
                clean = self._strip_unsupported({**request, "model": parsed.model}, adapter)
                clean.pop("models", None)
                clean.pop("route", None)
                clean.pop("provider", None)

                start = time.monotonic()
                result = await adapter.send_request(clean)
                elapsed = time.monotonic() - start
                self._record_stats(result, parsed.provider, elapsed, streamed=False)
                return result
            except AnyModelError as e:
                last_error = e
                if e.code == 429:
                    self._rate_limiter.record(parsed.provider, retry_after=5.0)
                continue
            except Exception as e:
                last_error = e
                continue

        raise last_error or AnyModelError(502, "All models failed.")

    async def _stream_with_fallback(
        self, request: dict[str, Any], models: list[str],
    ) -> AsyncIterator[dict[str, Any]]:
        """Try streaming models in order until one succeeds."""
        models = self._apply_provider_preferences(models, request.get("provider"))
        last_error: Exception | None = None

        for model in models:
            try:
                parsed = parse_model_string(model, self._aliases)

                if self._rate_limiter.is_rate_limited(parsed.provider):
                    continue

                adapter = self._registry.get(parsed.provider)
                clean = self._strip_unsupported({**request, "model": parsed.model}, adapter)
                clean.pop("models", None)
                clean.pop("route", None)
                clean.pop("provider", None)

                return await adapter.send_streaming_request(clean)
            except AnyModelError as e:
                last_error = e
                if e.code == 429:
                    self._rate_limiter.record(parsed.provider, retry_after=5.0)
                continue
            except Exception as e:
                last_error = e
                continue

        raise last_error or AnyModelError(502, "All models failed.")

    def _apply_defaults(self, request: dict[str, Any]) -> dict[str, Any]:
        """Apply default settings to a request."""
        result = dict(request)
        for key in ("temperature", "max_tokens"):
            if key not in result and key in self._defaults:
                result[key] = self._defaults[key]
        return result

    def _strip_unsupported(self, request: dict[str, Any], adapter: Any) -> dict[str, Any]:
        """Remove parameters not supported by the provider."""
        result = dict(request)
        to_remove = []
        skip_keys = {"model", "messages", "stream", "models", "route", "transforms", "provider"}
        for key in result:
            if key in skip_keys:
                continue
            if not adapter.supports_parameter(key):
                to_remove.append(key)
        for key in to_remove:
            del result[key]
        return result

    def _apply_provider_preferences(
        self, models: list[str], prefs: dict[str, Any] | None,
    ) -> list[str]:
        """Filter and reorder models based on provider preferences."""
        if not prefs:
            return models

        filtered = list(models)

        # Apply ignore list
        ignore = set(prefs.get("ignore", []))
        if ignore:
            filtered = [
                m for m in filtered
                if parse_model_string(m, self._aliases).provider not in ignore
            ]

        # Apply only list
        only = set(prefs.get("only", []))
        if only:
            filtered = [
                m for m in filtered
                if parse_model_string(m, self._aliases).provider in only
            ]

        # Apply order
        order = prefs.get("order")
        if order:
            order_map = {p: i for i, p in enumerate(order)}
            filtered.sort(
                key=lambda m: order_map.get(parse_model_string(m, self._aliases).provider, 999)
            )

        return filtered

    def _record_stats(
        self, result: dict[str, Any], provider: str, elapsed: float, *, streamed: bool,
    ) -> None:
        """Record generation stats."""
        usage = result.get("usage", {})
        self._stats_store.record({
            "id": result.get("id", generate_id()),
            "model": result.get("model", ""),
            "provider_name": provider,
            "total_cost": 0,
            "tokens_prompt": usage.get("prompt_tokens", 0),
            "tokens_completion": usage.get("completion_tokens", 0),
            "latency": elapsed,
            "generation_time": elapsed,
            "created_at": str(int(time.time())),
            "finish_reason": result.get("choices", [{}])[0].get("finish_reason", "stop"),
            "streamed": streamed,
        })
