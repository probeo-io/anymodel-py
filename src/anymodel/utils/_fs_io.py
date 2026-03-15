"""Concurrency-limited async filesystem helpers.

Based on the Node anymodel's fs-io.ts. Provides queued reads/writes,
atomic durable writes, and directory caching.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import aiofiles

# ─── Concurrency Semaphores ──────────────────────────────────────────────────

_read_semaphore = asyncio.Semaphore(20)
_write_semaphore = asyncio.Semaphore(10)
_dir_cache: set[str] = set()


def configure_fs_io(
    *,
    read_concurrency: int | None = None,
    write_concurrency: int | None = None,
) -> None:
    """Adjust read/write concurrency limits at runtime."""
    global _read_semaphore, _write_semaphore
    if read_concurrency is not None:
        _read_semaphore = asyncio.Semaphore(read_concurrency)
    if write_concurrency is not None:
        _write_semaphore = asyncio.Semaphore(write_concurrency)


def get_fs_queue_status() -> dict[str, Any]:
    """Return current semaphore state (for diagnostics)."""
    return {
        "read_available": _read_semaphore._value,
        "write_available": _write_semaphore._value,
    }


async def wait_for_fs_queues_idle() -> None:
    """Wait until all queued operations complete (best-effort)."""
    await asyncio.sleep(0)


# ─── Directory Helpers ───────────────────────────────────────────────────────


async def ensure_dir(dir_path: str | Path) -> None:
    """Create directory if it doesn't exist (cached)."""
    key = str(dir_path)
    if key in _dir_cache:
        return
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    _dir_cache.add(key)


# ─── Read Operations ────────────────────────────────────────────────────────


async def read_file_queued(file_path: str | Path, encoding: str = "utf-8") -> str:
    """Read a file with concurrency limiting."""
    async with _read_semaphore:
        async with aiofiles.open(file_path, encoding=encoding) as f:
            return await f.read()


async def read_json_queued(file_path: str | Path) -> Any:
    """Read and parse a JSON file with concurrency limiting."""
    raw = await read_file_queued(file_path)
    return json.loads(raw)


async def read_dir_queued(dir_path: str | Path) -> list[str]:
    """List directory entries with concurrency limiting."""
    async with _read_semaphore:
        return os.listdir(dir_path)


async def path_exists_queued(file_path: str | Path) -> bool:
    """Check if a path exists with concurrency limiting."""
    async with _read_semaphore:
        return Path(file_path).exists()


# ─── Write Operations ───────────────────────────────────────────────────────


async def write_file_queued(file_path: str | Path, data: str) -> None:
    """Write a file with concurrency limiting."""
    async with _write_semaphore:
        await ensure_dir(Path(file_path).parent)
        async with aiofiles.open(file_path, mode="w", encoding="utf-8") as f:
            await f.write(data)


async def write_file_flushed_queued(file_path: str | Path, data: str) -> None:
    """Atomically write a file with fsync (temp file + rename).

    Ensures data hits disk before the file appears at its final path.
    """
    async with _write_semaphore:
        target = Path(file_path)
        await ensure_dir(target.parent)

        fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
        try:
            async with aiofiles.open(fd, mode="w", encoding="utf-8", closefd=True) as f:
                await f.write(data)
                await f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(target))
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


async def append_file_queued(file_path: str | Path, data: str) -> None:
    """Append to a file with concurrency limiting."""
    async with _write_semaphore:
        await ensure_dir(Path(file_path).parent)
        async with aiofiles.open(file_path, mode="a", encoding="utf-8") as f:
            await f.write(data)
