"""Tests for prompt cache key helpers."""

from __future__ import annotations

from anymodel._cache import create_prompt_cache_key, with_prompt_cache


def test_create_prompt_cache_key_is_deterministic() -> None:
    parts = {"workflow_version": 3, "domain": "bizee.com", "archetype_set": ["a", "b"]}
    assert create_prompt_cache_key(parts) == create_prompt_cache_key(parts)


def test_create_prompt_cache_key_is_order_independent() -> None:
    a = create_prompt_cache_key({"workflow_version": 3, "domain": "bizee.com"})
    b = create_prompt_cache_key({"domain": "bizee.com", "workflow_version": 3})
    assert a == b


def test_create_prompt_cache_key_differs_for_different_inputs() -> None:
    a = create_prompt_cache_key({"domain": "bizee.com"})
    b = create_prompt_cache_key({"domain": "other.com"})
    assert a != b


def test_create_prompt_cache_key_applies_prefix() -> None:
    key = create_prompt_cache_key({"domain": "bizee.com"}, prefix="workflow")
    assert key.startswith("workflow:")


def test_create_prompt_cache_key_respects_max_length() -> None:
    key = create_prompt_cache_key({"domain": "bizee.com"}, prefix="workflow", max_length=16)
    assert len(key) <= 16


def test_with_prompt_cache_attaches_cache_options() -> None:
    request = {"model": "gpt-5.6-luna", "messages": []}
    result = with_prompt_cache(request, {"key": "abc", "ttl": "1h"})
    assert result["cache"] == {"key": "abc", "ttl": "1h"}
    assert "cache" not in request  # original request untouched


def test_with_prompt_cache_merges_existing_cache_options() -> None:
    request = {"model": "gpt-5.6-luna", "messages": [], "cache": {"key": "abc"}}
    result = with_prompt_cache(request, {"ttl": "24h"})
    assert result["cache"] == {"key": "abc", "ttl": "24h"}
