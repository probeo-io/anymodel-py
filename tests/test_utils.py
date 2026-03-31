"""Tests for ID generation and model parser edge cases.

Note: parse_model_string basics and validate_request basics are covered
in test_model_parser.py and test_validate.py respectively. This file
covers additional cases from the TypeScript utils.test.ts that are not
already covered.
"""

from __future__ import annotations

import pytest

from anymodel._types import AnyModelError
from anymodel.utils._id import generate_id
from anymodel.utils._model_parser import parse_model_string
from anymodel.utils._validate import validate_request


# ─── generateId ─────────────────────────────────────────────────────────────

def test_generates_gen_prefixed_ids() -> None:
    gen_id = generate_id()
    assert gen_id.startswith("gen-")


def test_generates_unique_ids() -> None:
    ids = {generate_id() for _ in range(100)}
    assert len(ids) == 100


def test_supports_custom_prefix() -> None:
    gen_id = generate_id("batch")
    assert gen_id.startswith("batch-")


def test_has_sufficient_randomness() -> None:
    gen_id = generate_id()
    random_part = gen_id[len("gen-"):]
    assert len(random_part) >= 12


# ─── parseModelString (additional edge cases) ───────────────────────────────

def test_parse_provider_model_format() -> None:
    result = parse_model_string("anthropic/claude-sonnet-4-6")
    assert result.provider == "anthropic"
    assert result.model == "claude-sonnet-4-6"


def test_parse_models_with_slashes_in_name() -> None:
    result = parse_model_string("custom/meta-llama/llama-3.3-70b")
    assert result.provider == "custom"
    assert result.model == "meta-llama/llama-3.3-70b"


def test_parse_resolves_aliases() -> None:
    aliases = {"default": "anthropic/claude-sonnet-4-6", "fast": "anthropic/claude-haiku-4-5"}
    result = parse_model_string("default", aliases)
    assert result.provider == "anthropic"
    assert result.model == "claude-sonnet-4-6"


def test_parse_throws_on_missing_slash() -> None:
    with pytest.raises(AnyModelError, match="provider/model"):
        parse_model_string("justmodelname")


def test_parse_throws_on_empty_provider() -> None:
    with pytest.raises(AnyModelError):
        parse_model_string("/model")


def test_parse_throws_on_empty_model() -> None:
    with pytest.raises(AnyModelError):
        parse_model_string("provider/")


# ─── validateRequest (additional edge cases) ─────────────────────────────────

def test_validate_passes_valid_request() -> None:
    validate_request({
        "model": "anthropic/claude-sonnet-4-6",
        "messages": [{"role": "user", "content": "Hello"}],
    })


def test_validate_throws_on_missing_model() -> None:
    with pytest.raises(AnyModelError):
        validate_request({
            "model": "",
            "messages": [{"role": "user", "content": "Hello"}],
        })


def test_validate_throws_on_missing_messages() -> None:
    with pytest.raises(AnyModelError):
        validate_request({
            "model": "anthropic/claude-sonnet-4-6",
            "messages": [],
        })


def test_validate_throws_on_invalid_temperature() -> None:
    with pytest.raises(AnyModelError, match="temperature"):
        validate_request({
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 3.0,
        })
    with pytest.raises(AnyModelError, match="temperature"):
        validate_request({
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": -1.0,
        })


def test_validate_throws_on_invalid_top_p() -> None:
    with pytest.raises(AnyModelError, match="top_p"):
        validate_request({
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "top_p": 1.5,
        })


def test_validate_throws_on_too_many_stop_sequences() -> None:
    with pytest.raises(AnyModelError, match="stop"):
        validate_request({
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": ["a", "b", "c", "d", "e"],
        })


def test_validate_passes_with_models_array() -> None:
    validate_request({
        "model": "",
        "models": ["anthropic/claude-sonnet-4-6"],
        "route": "fallback",
        "messages": [{"role": "user", "content": "Hi"}],
    })
