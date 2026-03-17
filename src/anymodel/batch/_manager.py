"""Batch processing manager — native and concurrent."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from anymodel._types import AnyModelError
from anymodel.batch._store import BatchStore
from anymodel.providers._adapter import BatchAdapter
from anymodel.utils._id import generate_id
from anymodel.utils._model_parser import parse_model_string


class BatchManager:
    """Manages batch processing with native and concurrent paths."""

    def __init__(
        self,
        router: Any,
        *,
        dir: str | None = None,
        concurrency: int = 5,
        poll_interval: float = 5.0,
        aliases: dict[str, str] | None = None,
    ) -> None:
        self._store = BatchStore(dir)
        self._router = router
        self._concurrency = concurrency
        self._default_poll_interval = poll_interval
        self._aliases = aliases or {}
        self._batch_adapters: dict[str, BatchAdapter] = {}

    def register_batch_adapter(self, provider_name: str, adapter: BatchAdapter) -> None:
        """Register a native batch adapter for a provider."""
        self._batch_adapters[provider_name] = adapter

    def _get_native_adapter(self, model: str) -> BatchAdapter | None:
        """Check if the model's provider has a native batch adapter."""
        parsed = parse_model_string(model, self._aliases)
        return self._batch_adapters.get(parsed.provider)

    async def create(self, request: dict[str, Any]) -> dict[str, Any]:
        """Create a batch. Routes to native or concurrent processing."""
        model = request["model"]
        requests = request["requests"]
        parsed = parse_model_string(model, self._aliases)
        adapter = self._get_native_adapter(model)

        batch_id = generate_id("batch")
        now = datetime.now(timezone.utc).isoformat()
        batch_mode = "native" if adapter else "concurrent"

        batch: dict[str, Any] = {
            "id": batch_id,
            "object": "batch",
            "status": "pending",
            "model": model,
            "provider_name": parsed.provider,
            "batch_mode": batch_mode,
            "total": len(requests),
            "completed": 0,
            "failed": 0,
            "created_at": now,
            "completed_at": None,
            "expires_at": None,
        }

        await self._store.create(batch)
        await self._store.save_requests(batch_id, requests)

        if adapter:
            await self._process_native_batch(batch, adapter, requests, request.get("options"))
        else:
            # Fire off concurrent processing in the background — streams from disk
            asyncio.create_task(self._process_concurrent_batch(batch, request.get("options")))

        return batch

    async def create_and_poll(
        self,
        request: dict[str, Any],
        *,
        interval: float | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Create a batch and poll until completion."""
        batch = await self.create(request)
        return await self.poll(batch["id"], interval=interval, on_progress=on_progress)

    async def poll(
        self,
        batch_id: str,
        *,
        interval: float | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Poll a batch until completion."""
        poll_interval = interval or self._default_poll_interval

        while True:
            batch = await self._store.get_meta(batch_id)
            if batch is None:
                raise AnyModelError(404, f"Batch {batch_id} not found.")

            # Sync native batch status if applicable
            if batch["batch_mode"] == "native" and batch["status"] in ("pending", "processing"):
                await self._sync_native_batch_status(batch)
                batch = await self._store.get_meta(batch_id)
                if batch is None:
                    raise AnyModelError(404, f"Batch {batch_id} not found.")

            if on_progress:
                on_progress(batch)

            if batch["status"] in ("completed", "failed", "cancelled"):
                results = await self._store.get_results(batch_id)
                usage = {"total_prompt_tokens": 0, "total_completion_tokens": 0, "estimated_cost": 0}
                for r in results:
                    resp = r.get("response")
                    if resp and resp.get("usage"):
                        usage["total_prompt_tokens"] += resp["usage"].get("prompt_tokens", 0)
                        usage["total_completion_tokens"] += resp["usage"].get("completion_tokens", 0)
                return {
                    "id": batch_id,
                    "status": batch["status"],
                    "results": results,
                    "usage_summary": usage,
                }

            await asyncio.sleep(poll_interval)

    async def get(self, batch_id: str) -> dict[str, Any] | None:
        """Get batch metadata."""
        return await self._store.get_meta(batch_id)

    async def list(self) -> list[dict[str, Any]]:
        """List all batches."""
        ids = await self._store.list_batches()
        batches = []
        for bid in ids:
            meta = await self._store.get_meta(bid)
            if meta:
                batches.append(meta)
        return batches

    async def results(self, batch_id: str) -> dict[str, Any]:
        """Get batch results."""
        batch = await self._store.get_meta(batch_id)
        if batch is None:
            raise AnyModelError(404, f"Batch {batch_id} not found.")

        result_items = await self._store.get_results(batch_id)
        usage = {"total_prompt_tokens": 0, "total_completion_tokens": 0, "estimated_cost": 0}
        for r in result_items:
            resp = r.get("response")
            if resp and resp.get("usage"):
                usage["total_prompt_tokens"] += resp["usage"].get("prompt_tokens", 0)
                usage["total_completion_tokens"] += resp["usage"].get("completion_tokens", 0)

        return {
            "id": batch_id,
            "status": batch["status"],
            "results": result_items,
            "usage_summary": usage,
        }

    async def cancel(self, batch_id: str) -> dict[str, Any]:
        """Cancel a batch."""
        batch = await self._store.get_meta(batch_id)
        if batch is None:
            raise AnyModelError(404, f"Batch {batch_id} not found.")

        # Cancel at provider for native batches
        if batch["batch_mode"] == "native":
            adapter = self._batch_adapters.get(batch["provider_name"])
            if adapter:
                provider_state = await self._store.load_provider_state(batch_id)
                if provider_state and provider_state.get("providerBatchId"):
                    try:
                        await adapter.cancel_batch(provider_state["providerBatchId"])
                    except Exception:
                        pass  # Best-effort cancellation

        batch["status"] = "cancelled"
        batch["completed_at"] = datetime.now(timezone.utc).isoformat()
        await self._store.update_meta(batch)
        return batch

    async def _process_native_batch(
        self,
        batch: dict[str, Any],
        adapter: BatchAdapter,
        requests: list[dict[str, Any]],
        options: dict[str, Any] | None,
    ) -> None:
        """Submit batch to native provider API."""
        try:
            parsed = parse_model_string(batch["model"], self._aliases)
            result = await adapter.create_batch(parsed.model, requests, options)

            await self._store.save_provider_state(batch["id"], {
                "providerBatchId": result["providerBatchId"],
                "metadata": result.get("metadata"),
            })

            batch["status"] = "processing"
            await self._store.update_meta(batch)

        except Exception:
            batch["status"] = "failed"
            batch["completed_at"] = datetime.now(timezone.utc).isoformat()
            await self._store.update_meta(batch)
            raise

    async def _sync_native_batch_status(self, batch: dict[str, Any]) -> None:
        """Poll native provider and sync status."""
        adapter = self._batch_adapters.get(batch["provider_name"])
        if not adapter:
            return

        provider_state = await self._store.load_provider_state(batch["id"])
        if not provider_state or not provider_state.get("providerBatchId"):
            return

        provider_id = provider_state["providerBatchId"]
        status = await adapter.poll_batch(provider_id)

        batch["completed"] = status.completed
        batch["failed"] = status.failed

        if status.status == "completed":
            # Download results
            results = await adapter.get_batch_results(provider_id)
            for r in results:
                await self._store.append_result(batch["id"], r)

            batch["status"] = "completed"
            batch["completed_at"] = datetime.now(timezone.utc).isoformat()
        elif status.status == "failed":
            batch["status"] = "failed"
            batch["completed_at"] = datetime.now(timezone.utc).isoformat()
        elif status.status == "cancelled":
            batch["status"] = "cancelled"
            batch["completed_at"] = datetime.now(timezone.utc).isoformat()
        else:
            batch["status"] = "processing"

        await self._store.update_meta(batch)

    async def _process_concurrent_batch(
        self,
        batch: dict[str, Any],
        options: dict[str, Any] | None,
    ) -> None:
        """Process batch with concurrent requests. Streams from disk."""
        batch["status"] = "processing"
        await self._store.update_meta(batch)

        semaphore = asyncio.Semaphore(self._concurrency)
        active: set[asyncio.Task[None]] = set()

        async def process_one(req: dict[str, Any]) -> None:
            async with semaphore:
                custom_id = req.get("custom_id", generate_id("req"))
                try:
                    chat_request: dict[str, Any] = {
                        "model": batch["model"],
                        "messages": req["messages"],
                    }
                    if options:
                        chat_request.update(options)
                    for key in ("max_tokens", "temperature", "top_p", "top_k", "stop",
                                "response_format", "tools", "tool_choice", "service_tier"):
                        if key in req:
                            chat_request[key] = req[key]

                    result = await self._router.complete(chat_request)

                    await self._store.append_result(batch["id"], {
                        "custom_id": custom_id,
                        "status": "success",
                        "response": result,
                        "error": None,
                    })
                    batch["completed"] += 1
                except Exception as e:
                    code = getattr(e, "code", 500)
                    await self._store.append_result(batch["id"], {
                        "custom_id": custom_id,
                        "status": "error",
                        "response": None,
                        "error": {"code": code, "message": str(e)},
                    })
                    batch["failed"] += 1

                await self._store.update_meta(batch)

        # Stream requests from disk instead of holding all in memory
        async for req in self._store.stream_requests(batch["id"]):
            meta = await self._store.get_meta(batch["id"])
            if meta and meta["status"] == "cancelled":
                break

            # Wait for a slot if at concurrency limit
            if len(active) >= self._concurrency:
                done, active_set = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                active = active_set

            task = asyncio.create_task(process_one(req))
            active.add(task)

        if active:
            await asyncio.wait(active)

        batch = await self._store.get_meta(batch["id"]) or batch
        if batch["status"] != "cancelled":
            batch["status"] = "completed" if batch["failed"] < batch["total"] else "failed"
            batch["completed_at"] = datetime.now(timezone.utc).isoformat()
            await self._store.update_meta(batch)
