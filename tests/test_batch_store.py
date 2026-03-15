"""Tests for BatchStore."""

import pytest

from anymodel.batch._store import BatchStore


def _make_batch(batch_id: str = "batch-test1", **kwargs):
    base = {
        "id": batch_id,
        "object": "batch",
        "status": "pending",
        "model": "openai/gpt-4o",
        "provider_name": "openai",
        "batch_mode": "concurrent",
        "total": 3,
        "completed": 0,
        "failed": 0,
        "created_at": "2026-03-15T00:00:00Z",
        "completed_at": None,
        "expires_at": None,
    }
    base.update(kwargs)
    return base


@pytest.mark.asyncio
async def test_create_and_retrieve(tmp_batch_dir):
    store = BatchStore(str(tmp_batch_dir))
    batch = _make_batch()
    await store.create(batch)
    result = await store.get_meta("batch-test1")
    assert result == batch


@pytest.mark.asyncio
async def test_update_meta(tmp_batch_dir):
    store = BatchStore(str(tmp_batch_dir))
    batch = _make_batch("batch-test2")
    await store.create(batch)
    batch["status"] = "completed"
    batch["completed"] = 3
    await store.update_meta(batch)
    result = await store.get_meta("batch-test2")
    assert result["status"] == "completed"
    assert result["completed"] == 3


@pytest.mark.asyncio
async def test_append_and_get_results(tmp_batch_dir):
    store = BatchStore(str(tmp_batch_dir))
    await store.create(_make_batch("batch-test3"))

    await store.append_result("batch-test3", {
        "custom_id": "req-1", "status": "success",
        "response": {"id": "gen-abc", "choices": []}, "error": None,
    })
    await store.append_result("batch-test3", {
        "custom_id": "req-2", "status": "error",
        "response": None, "error": {"code": 429, "message": "Rate limited"},
    })

    results = await store.get_results("batch-test3")
    assert len(results) == 2
    assert results[0]["custom_id"] == "req-1"
    assert results[1]["status"] == "error"


@pytest.mark.asyncio
async def test_list_batches(tmp_batch_dir):
    store = BatchStore(str(tmp_batch_dir))
    for bid in ["batch-a", "batch-b", "batch-c"]:
        await store.create(_make_batch(bid))
    batches = await store.list_batches()
    assert len(batches) == 3
    assert "batch-a" in batches


@pytest.mark.asyncio
async def test_provider_state(tmp_batch_dir):
    store = BatchStore(str(tmp_batch_dir))
    await store.create(_make_batch("batch-ps"))
    await store.save_provider_state("batch-ps", {"providerBatchId": "oai-123"})
    state = await store.load_provider_state("batch-ps")
    assert state["providerBatchId"] == "oai-123"


@pytest.mark.asyncio
async def test_nonexistent_batch(tmp_batch_dir):
    store = BatchStore(str(tmp_batch_dir))
    assert await store.get_meta("nonexistent") is None
