"""Generation stats tracking."""

from __future__ import annotations

from typing import Any


class GenerationStatsStore:
    """In-memory store for generation statistics, keyed by generation ID."""

    def __init__(self, max_entries: int = 1000) -> None:
        self._store: dict[str, dict[str, Any]] = {}
        self._max_entries = max_entries

    def record(self, stats: dict[str, Any]) -> None:
        """Record generation stats."""
        gen_id = stats.get("id")
        if not gen_id:
            return

        # Evict oldest if at capacity
        if len(self._store) >= self._max_entries:
            oldest = next(iter(self._store))
            del self._store[oldest]

        self._store[gen_id] = stats

    def get(self, gen_id: str) -> dict[str, Any] | None:
        """Retrieve stats for a generation ID."""
        return self._store.get(gen_id)

    def list(self) -> list[dict[str, Any]]:
        """List all recorded stats."""
        return list(self._store.values())
