"""Tests for types module."""

from anymodel._types import AnyModelError


def test_anymodel_error_creation():
    err = AnyModelError(429, "Rate limited", {"provider_name": "openai"})
    assert err.code == 429
    assert str(err) == "Rate limited"
    assert err.metadata["provider_name"] == "openai"


def test_anymodel_error_to_dict():
    err = AnyModelError(400, "Bad request")
    result = err.to_dict()
    assert result["error"]["code"] == 400
    assert result["error"]["message"] == "Bad request"
    assert result["error"]["metadata"] == {}


def test_anymodel_error_is_exception():
    err = AnyModelError(500, "Server error")
    assert isinstance(err, Exception)
