"""Tests for model parser."""

import pytest

from anymodel._types import AnyModelError
from anymodel.utils._model_parser import parse_model_string


def test_parse_basic():
    result = parse_model_string("openai/gpt-4o")
    assert result.provider == "openai"
    assert result.model == "gpt-4o"


def test_parse_with_slashes():
    result = parse_model_string("together/meta-llama/Llama-3.3-70B")
    assert result.provider == "together"
    assert result.model == "meta-llama/Llama-3.3-70B"


def test_parse_alias():
    result = parse_model_string("fast", {"fast": "anthropic/claude-haiku-4-5"})
    assert result.provider == "anthropic"
    assert result.model == "claude-haiku-4-5"


def test_parse_invalid():
    with pytest.raises(AnyModelError) as exc:
        parse_model_string("no-slash")
    assert exc.value.code == 400
