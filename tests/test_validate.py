"""Tests for request validation."""

import pytest

from anymodel._types import AnyModelError
from anymodel.utils._validate import validate_request


def test_valid_request():
    validate_request({"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "Hi"}]})


def test_missing_model():
    with pytest.raises(AnyModelError):
        validate_request({"messages": [{"role": "user", "content": "Hi"}]})


def test_missing_messages():
    with pytest.raises(AnyModelError):
        validate_request({"model": "openai/gpt-4o"})


def test_invalid_temperature():
    with pytest.raises(AnyModelError):
        validate_request({"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "Hi"}], "temperature": 3.0})


def test_invalid_stop_length():
    with pytest.raises(AnyModelError):
        validate_request({
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": ["a", "b", "c", "d", "e"],
        })


def test_models_array_valid():
    validate_request({"models": ["openai/gpt-4o"], "model": "", "messages": [{"role": "user", "content": "Hi"}]})
