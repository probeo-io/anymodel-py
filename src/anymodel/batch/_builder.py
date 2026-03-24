"""Ergonomic batch builder — add prompts one at a time, submit when ready."""

from __future__ import annotations

from typing import Any

from anymodel._types import AnyModelError
from anymodel.utils._id import generate_id

try:
    from anymodel.generated.pricing import calculate_cost as _calculate_cost
except ImportError:

    def _calculate_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:  # type: ignore[misc]
        return 0.0


# ─── Constants ────────────────────────────────────────────────────────────────

RETRYABLE_CODES = frozenset({408, 429, 500, 502, 503, 529})


# ─── BatchBuilder ─────────────────────────────────────────────────────────────


class BatchBuilder:
    """
    Ergonomic batch builder. Add prompts one at a time, submit when ready.

    Example::

        batch = client.batches.open({"model": "anthropic/claude-sonnet-4-6", "system": "You are an expert."})
        batch.add("What is an LLC?")
        batch.add("How do I dissolve an LLC?")
        await batch.submit()
        results = await batch.poll()
    """

    def __init__(self, config: dict[str, Any], store: Any, manager: Any) -> None:
        from anymodel.batch._store import BatchStore
        from anymodel.batch._manager import BatchManager

        self._batch_id: str = generate_id("batch")
        self._config: dict[str, Any] = config
        self._store: BatchStore = store
        self._manager: BatchManager = manager
        self._count: int = 0
        self._submitted: bool = False

    @property
    def id(self) -> str:
        """The batch ID (available immediately after construction)."""
        return self._batch_id

    @property
    def size(self) -> int:
        """Number of prompts added so far."""
        return self._count

    def add(self, content: str | list[dict[str, Any]]) -> "BatchBuilder":
        """
        Add a prompt to the batch. Written to disk immediately.

        Args:
            content: A string prompt or a list of message dicts for multi-turn.
        """
        if self._submitted:
            raise AnyModelError(400, "Cannot add to a submitted batch. Use retry() for failed items.")

        messages: list[dict[str, Any]]

        if isinstance(content, str):
            messages = []
            system = self._config.get("system")
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": content})
        else:
            messages = list(content)
            # Prepend system if configured and not already present
            system = self._config.get("system")
            if system and not any(m.get("role") == "system" for m in messages):
                messages.insert(0, {"role": "system", "content": system})

        custom_id = f"req-{self._count:06d}"

        item: dict[str, Any] = {
            "custom_id": custom_id,
            "messages": messages,
        }

        # Apply batch-level options
        for key in (
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop",
            "response_format",
            "tools",
            "tool_choice",
            "service_tier",
        ):
            if key in self._config:
                item[key] = self._config[key]

        # Write to disk immediately (fire and forget — append_request is fast)
        # Use asyncio to schedule the coroutine if we're in an event loop,
        # otherwise just use the sync path
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._store.append_request(self._batch_id, item))
        except RuntimeError:
            # No running event loop — run synchronously
            asyncio.run(self._store.append_request(self._batch_id, item))

        self._count += 1
        return self

    async def submit(self) -> str:
        """
        Submit the batch for processing.
        Reads prompts from disk and dispatches to the provider.
        """
        if self._submitted:
            raise AnyModelError(400, "Batch already submitted.")
        if self._count == 0:
            raise AnyModelError(400, "Cannot submit an empty batch. Call add() first.")

        # Collect requests from disk
        requests: list[dict[str, Any]] = []
        async for item in self._store.stream_requests(self._batch_id):
            requests.append(item)

        create_request: dict[str, Any] = {
            "model": self._config["model"],
            "requests": requests,
        }

        batch_mode = self._config.get("batch_mode")
        if batch_mode:
            create_request["batch_mode"] = batch_mode

        # Use the manager's create — it will save requests and dispatch
        await self._manager.create(create_request)

        self._submitted = True
        return self._batch_id

    async def poll(self, **kwargs: Any) -> dict[str, Any]:
        """
        Poll until the batch completes. Returns clean succeeded/failed results.
        """
        if not self._submitted:
            raise AnyModelError(400, "Batch not yet submitted. Call submit() first.")

        raw = await self._manager.poll(self._batch_id, **kwargs)
        return await self._transform_results(raw)

    async def get_results(self) -> dict[str, Any]:
        """
        Get results for an already-completed batch (non-blocking).
        """
        if not self._submitted:
            raise AnyModelError(400, "Batch not yet submitted. Call submit() first.")

        raw = await self._manager.results(self._batch_id)
        return await self._transform_results(raw)

    def retry(self, failed: list[dict[str, Any]]) -> "BatchBuilder":
        """
        Create a new batch builder pre-loaded with the failed items from a previous batch.
        Call submit() on the returned builder to retry.
        """
        retry_builder = BatchBuilder(self._config, self._store, self._manager)
        for item in failed:
            retry_builder.add(item["prompt"])
        return retry_builder

    async def cancel(self) -> None:
        """Cancel the batch."""
        await self._manager.cancel(self._batch_id)

    # ─── Internal ─────────────────────────────────────────────────────────────

    async def _transform_results(
        self,
        raw: dict[str, Any],
    ) -> dict[str, Any]:
        """Transform raw batch results into the clean succeeded/failed format."""
        # Build a map of custom_id -> original prompt
        prompt_map: dict[str, str | list[dict[str, Any]]] = {}
        async for item in self._store.stream_requests(self._batch_id):
            # Extract the user's original prompt from messages
            user_messages = [m for m in item["messages"] if m.get("role") != "system"]
            if len(user_messages) == 1 and isinstance(user_messages[0].get("content"), str):
                prompt_map[item["custom_id"]] = user_messages[0]["content"]
            else:
                prompt_map[item["custom_id"]] = item["messages"]

        succeeded: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0.0

        for result in raw.get("results", []):
            if result.get("status") == "success" and result.get("response"):
                response = result["response"]
                usage = response.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                cost = _calculate_cost(
                    response.get("model") or self._config["model"],
                    prompt_tokens,
                    completion_tokens,
                )

                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_cost += cost

                succeeded.append({
                    "id": result["custom_id"],
                    "content": (
                        response.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    if response.get("choices")
                    else "",
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                    "cost": cost,
                    "raw": response,
                })
            else:
                error_code = (result.get("error") or {}).get("code", 500)
                failed.append({
                    "id": result.get("custom_id", ""),
                    "prompt": prompt_map.get(result.get("custom_id", ""), ""),
                    "error": {
                        "code": error_code,
                        "message": (result.get("error") or {}).get("message", "Unknown error"),
                        "provider": self._config["model"].split("/")[0],
                    },
                    "retryable": error_code in RETRYABLE_CODES,
                })

        return {
            "id": self._batch_id,
            "succeeded": succeeded,
            "failed": failed,
            "usage": {
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "estimated_cost": total_cost,
            },
        }
