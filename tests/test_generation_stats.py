"""Tests for GenerationStatsStore."""

from anymodel.utils._generation_stats import GenerationStatsStore


def test_records_and_retrieves_a_generation() -> None:
    store = GenerationStatsStore()
    store.record({
        "id": "gen-abc123",
        "model": "anthropic/claude-sonnet-4-6",
        "provider_name": "anthropic",
        "tokens_prompt": 100,
        "tokens_completion": 50,
        "latency": 1.5,
        "streamed": False,
    })

    stats = store.get("gen-abc123")
    assert stats is not None
    assert stats["id"] == "gen-abc123"
    assert stats["model"] == "anthropic/claude-sonnet-4-6"
    assert stats["provider_name"] == "anthropic"
    assert stats["tokens_prompt"] == 100
    assert stats["tokens_completion"] == 50
    assert stats["streamed"] is False


def test_returns_none_for_unknown_id() -> None:
    store = GenerationStatsStore()
    assert store.get("nonexistent") is None


def test_lists_all_generations() -> None:
    store = GenerationStatsStore()
    for i in range(5):
        store.record({
            "id": f"gen-{i}",
            "model": "openai/gpt-4o",
            "provider_name": "openai",
            "tokens_prompt": 10,
            "tokens_completion": 5,
            "latency": 0.5,
            "streamed": False,
        })

    entries = store.list()
    assert len(entries) == 5


def test_evicts_oldest_when_at_capacity() -> None:
    store = GenerationStatsStore(max_entries=3)
    for i in range(5):
        store.record({
            "id": f"gen-{i}",
            "model": "openai/gpt-4o",
            "provider_name": "openai",
            "tokens_prompt": 10,
            "tokens_completion": 5,
            "latency": 0.1,
            "streamed": False,
        })

    assert store.get("gen-0") is None
    assert store.get("gen-1") is None
    assert store.get("gen-4") is not None
