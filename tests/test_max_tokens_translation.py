"""Tests for max_tokens -> max_completion_tokens translation."""

from anymodel.providers._openai import OpenAIAdapter
from anymodel.providers._anthropic import AnthropicAdapter
from anymodel.providers._google import GoogleAdapter


def _base_request(model: str, max_tokens: int | None = None) -> dict:
    req: dict = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    if max_tokens is not None:
        req["max_tokens"] = max_tokens
    return req


class TestOpenAIMaxTokens:
    """OpenAI adapter: max_tokens -> max_completion_tokens for newer models."""

    def setup_method(self):
        self.adapter = OpenAIAdapter("test-key")

    def test_gpt_4o(self):
        body = self.adapter._build_request_body(_base_request("gpt-4o", 1000))
        assert body.get("max_completion_tokens") == 1000
        assert "max_tokens" not in body

    def test_gpt_4o_mini(self):
        body = self.adapter._build_request_body(_base_request("gpt-4o-mini", 500))
        assert body.get("max_completion_tokens") == 500
        assert "max_tokens" not in body

    def test_o1(self):
        body = self.adapter._build_request_body(_base_request("o1", 2000))
        assert body.get("max_completion_tokens") == 2000
        assert "max_tokens" not in body

    def test_o3(self):
        body = self.adapter._build_request_body(_base_request("o3", 4000))
        assert body.get("max_completion_tokens") == 4000
        assert "max_tokens" not in body

    def test_o4_mini(self):
        body = self.adapter._build_request_body(_base_request("o4-mini", 8000))
        assert body.get("max_completion_tokens") == 8000
        assert "max_tokens" not in body

    def test_gpt_5_mini(self):
        body = self.adapter._build_request_body(_base_request("gpt-5-mini", 16000))
        assert body.get("max_completion_tokens") == 16000
        assert "max_tokens" not in body

    # Legacy models keep max_tokens

    def test_gpt_4_turbo(self):
        body = self.adapter._build_request_body(_base_request("gpt-4-turbo", 1000))
        assert body.get("max_tokens") == 1000
        assert "max_completion_tokens" not in body

    def test_gpt_35_turbo(self):
        body = self.adapter._build_request_body(_base_request("gpt-3.5-turbo", 500))
        assert body.get("max_tokens") == 500
        assert "max_completion_tokens" not in body

    # Omitted max_tokens

    def test_undefined_new_model(self):
        body = self.adapter._build_request_body(_base_request("gpt-4o"))
        assert "max_tokens" not in body
        assert "max_completion_tokens" not in body

    def test_undefined_legacy_model(self):
        body = self.adapter._build_request_body(_base_request("gpt-4-turbo"))
        assert "max_tokens" not in body
        assert "max_completion_tokens" not in body


class TestAnthropicMaxTokens:
    """Anthropic adapter: always sends max_tokens."""

    def setup_method(self):
        self.adapter = AnthropicAdapter("test-key")

    def test_always_max_tokens(self):
        body = self.adapter._translate_request(_base_request("claude-sonnet-4-6", 2000))
        assert body.get("max_tokens") == 2000
        assert "max_completion_tokens" not in body

    def test_defaults_to_4096(self):
        body = self.adapter._translate_request(_base_request("claude-sonnet-4-6"))
        assert body.get("max_tokens") == 4096


class TestGoogleMaxTokens:
    """Google adapter: translates to maxOutputTokens."""

    def setup_method(self):
        self.adapter = GoogleAdapter("test-key")

    def test_max_output_tokens(self):
        body = self.adapter._translate_request(_base_request("gemini-2.5-pro", 8000))
        assert body["generationConfig"]["maxOutputTokens"] == 8000
        assert "max_tokens" not in body
        assert "max_completion_tokens" not in body

    def test_omits_when_undefined(self):
        body = self.adapter._translate_request(_base_request("gemini-2.5-flash"))
        assert "generationConfig" not in body
