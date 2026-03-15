"""Disk-based batch persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anymodel.utils._fs_io import (
    append_file_queued,
    ensure_dir,
    path_exists_queued,
    read_dir_queued,
    read_file_queued,
    read_json_queued,
    write_file_flushed_queued,
    write_file_queued,
)


class BatchStore:
    """Persists batch metadata and results to disk."""

    def __init__(self, base_dir: str | None = None) -> None:
        self._base_dir = base_dir or str(Path.cwd() / ".anymodel" / "batches")
        self._initialized = False

    async def _init(self) -> None:
        if not self._initialized:
            await ensure_dir(self._base_dir)
            self._initialized = True

    def _batch_dir(self, batch_id: str) -> str:
        return str(Path(self._base_dir) / batch_id)

    async def create(self, batch: dict[str, Any]) -> None:
        """Create a new batch on disk."""
        await self._init()
        batch_dir = self._batch_dir(batch["id"])
        await ensure_dir(batch_dir)
        await write_file_flushed_queued(
            str(Path(batch_dir) / "meta.json"),
            json.dumps(batch, indent=2),
        )

    async def get_meta(self, batch_id: str) -> dict[str, Any] | None:
        """Read batch metadata."""
        meta_path = str(Path(self._batch_dir(batch_id)) / "meta.json")
        if not await path_exists_queued(meta_path):
            return None
        return await read_json_queued(meta_path)

    async def update_meta(self, batch: dict[str, Any]) -> None:
        """Update batch metadata atomically."""
        await write_file_flushed_queued(
            str(Path(self._batch_dir(batch["id"])) / "meta.json"),
            json.dumps(batch, indent=2),
        )

    async def save_requests(self, batch_id: str, requests: list[dict[str, Any]]) -> None:
        """Save batch requests as JSONL."""
        lines = [json.dumps(r) for r in requests]
        await write_file_queued(
            str(Path(self._batch_dir(batch_id)) / "requests.jsonl"),
            "\n".join(lines) + "\n",
        )

    async def append_result(self, batch_id: str, result: dict[str, Any]) -> None:
        """Append a single result to the results file."""
        await append_file_queued(
            str(Path(self._batch_dir(batch_id)) / "results.jsonl"),
            json.dumps(result) + "\n",
        )

    async def get_results(self, batch_id: str) -> list[dict[str, Any]]:
        """Read all results for a batch."""
        results_path = str(Path(self._batch_dir(batch_id)) / "results.jsonl")
        if not await path_exists_queued(results_path):
            return []
        raw = await read_file_queued(results_path)
        results = []
        for line in raw.strip().split("\n"):
            if line.strip():
                results.append(json.loads(line))
        return results

    async def save_provider_state(self, batch_id: str, state: dict[str, Any]) -> None:
        """Save provider-specific state (e.g., provider batch ID)."""
        await write_file_flushed_queued(
            str(Path(self._batch_dir(batch_id)) / "provider.json"),
            json.dumps(state, indent=2),
        )

    async def load_provider_state(self, batch_id: str) -> dict[str, Any] | None:
        """Load provider-specific state."""
        provider_path = str(Path(self._batch_dir(batch_id)) / "provider.json")
        if not await path_exists_queued(provider_path):
            return None
        return await read_json_queued(provider_path)

    async def list_batches(self) -> list[str]:
        """List all batch IDs on disk."""
        await self._init()
        try:
            entries = await read_dir_queued(self._base_dir)
            return [e for e in entries if not e.startswith(".")]
        except FileNotFoundError:
            return []
