"""Provider adapter protocols."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ProviderAdapter(Protocol):
    """Interface for LLM provider adapters."""

    @property
    def name(self) -> str: ...

    async def send_request(self, request: dict[str, Any]) -> dict[str, Any]: ...

    async def send_streaming_request(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]: ...

    async def list_models(self) -> list[dict[str, Any]]: ...

    def supports_parameter(self, param: str) -> bool: ...

    def supports_batch(self) -> bool: ...

    def translate_error(self, error: Exception) -> dict[str, Any]: ...


class NativeBatchStatus:
    """Status returned from polling a native batch."""

    def __init__(
        self,
        *,
        status: str,
        completed: int = 0,
        failed: int = 0,
        total: int = 0,
    ) -> None:
        self.status = status
        self.completed = completed
        self.failed = failed
        self.total = total


@runtime_checkable
class BatchAdapter(Protocol):
    """Interface for native batch processing."""

    async def create_batch(
        self,
        model: str,
        requests: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    async def poll_batch(self, provider_batch_id: str) -> NativeBatchStatus: ...

    async def get_batch_results(self, provider_batch_id: str) -> list[dict[str, Any]]: ...

    async def cancel_batch(self, provider_batch_id: str) -> None: ...
