"""OpenAI provider adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from anymodel._types import AnyModelError
from anymodel.utils._timeout import get_default_timeout, get_flex_timeout

OPENAI_API_BASE = "https://api.openai.com/v1"

SUPPORTED_PARAMS = frozenset({
    "temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty",
    "seed", "stop", "stream", "logprobs", "top_logprobs", "response_format",
    "tools", "tool_choice", "user", "logit_bias", "service_tier",
})


class OpenAIAdapter:
    """OpenAI chat completion adapter."""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = (base_url or OPENAI_API_BASE).rstrip("/")
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "openai"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                timeout=get_default_timeout(),
            )
        return self._client

    def _re_prefix_id(self, id: str) -> str:
        if id and id.startswith("chatcmpl-"):
            return f"gen-{id[9:]}"
        return id if id.startswith("gen-") else f"gen-{id}"

    def _map_error_code(self, status: int) -> int:
        if status in (401, 403):
            return 401
        if status == 429:
            return 429
        if status in (400, 422):
            return 400
        if status >= 500:
            return 502
        return status

    def _build_request_body(self, request: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request["model"],
            "messages": request["messages"],
        }
        for param in SUPPORTED_PARAMS:
            if param in request:
                body[param] = request[param]
        return body

    async def _make_request(
        self, path: str, body: dict[str, Any] | None = None, method: str = "POST",
        timeout: float | None = None,
    ) -> httpx.Response:
        client = self._get_client()
        kwargs: dict[str, Any] = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        if method == "GET":
            res = await client.get(path, **kwargs)
        else:
            res = await client.post(path, json=body, **kwargs)

        if res.status_code >= 400:
            try:
                error_body = res.json()
            except Exception:
                error_body = {"message": res.reason_phrase}
            msg = (
                error_body.get("error", {}).get("message")
                or error_body.get("message")
                or res.reason_phrase
            )
            raise AnyModelError(
                self._map_error_code(res.status_code),
                msg or "Unknown OpenAI error",
                {"provider_name": "openai", "raw": error_body},
            )
        return res

    def _request_timeout(self, request: dict[str, Any]) -> float | None:
        """Return a per-request timeout override for flex requests."""
        if request.get("service_tier") == "flex":
            return get_flex_timeout()
        return None

    async def send_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a non-streaming chat completion request."""
        body = self._build_request_body(request)
        timeout = self._request_timeout(request)
        res = await self._make_request("/chat/completions", body, timeout=timeout)
        data = res.json()
        return self._translate_response(data)

    async def send_streaming_request(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Send a streaming chat completion request."""
        body = self._build_request_body({**request, "stream": True})
        timeout = self._request_timeout(request)
        client = self._get_client()
        req = client.build_request("POST", "/chat/completions", json=body)
        res = await client.send(req, stream=True, timeout=timeout)

        if res.status_code >= 400:
            error_text = await res.aread()
            await res.aclose()
            try:
                error_body = json.loads(error_text)
            except Exception:
                error_body = {"message": res.reason_phrase}
            msg = error_body.get("error", {}).get("message") or res.reason_phrase
            raise AnyModelError(
                self._map_error_code(res.status_code),
                msg or "Unknown OpenAI error",
                {"provider_name": "openai", "raw": error_body},
            )

        return self._iter_sse(res)

    async def _iter_sse(self, res: httpx.Response) -> AsyncIterator[dict[str, Any]]:
        try:
            async for line in res.aiter_lines():
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                if line == "data: [DONE]":
                    return
                if line.startswith("data: "):
                    chunk = json.loads(line[6:])
                    chunk["id"] = self._re_prefix_id(chunk.get("id", ""))
                    chunk["model"] = f"openai/{chunk.get('model', '')}"
                    yield chunk
        finally:
            await res.aclose()

    def _translate_response(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": self._re_prefix_id(data.get("id", "")),
            "object": "chat.completion",
            "created": data.get("created", 0),
            "model": f"openai/{data.get('model', '')}",
            "choices": data.get("choices", []),
            "usage": data.get("usage", {}),
        }

    async def list_models(self) -> list[dict[str, Any]]:
        """Fetch available chat models from OpenAI."""
        try:
            res = await self._make_request("/models", method="GET")
            data = res.json()
        except Exception:
            return []

        models = []
        for m in data.get("data", []):
            mid = m.get("id", "")
            # Exclude non-chat models
            skip_patterns = ("embedding", "whisper", "tts", "dall-e", "davinci", "babbage", "moderation", "realtime")
            if any(p in mid for p in skip_patterns):
                continue
            if mid.startswith("ft:"):
                continue
            if not (mid.startswith("gpt-") or mid.startswith("o1") or mid.startswith("o3") or mid.startswith("o4") or mid.startswith("chatgpt-")):
                continue

            models.append({
                "id": f"openai/{mid}",
                "name": mid,
                "created": m.get("created", 0),
                "description": "",
                "context_length": 128000,
                "pricing": {"prompt": "0", "completion": "0"},
                "architecture": {
                    "modality": "text+image->text",
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                    "tokenizer": "o200k_base",
                },
                "top_provider": {
                    "context_length": 128000,
                    "max_completion_tokens": 16384,
                    "is_moderated": True,
                },
                "supported_parameters": list(SUPPORTED_PARAMS),
            })
        return models

    def supports_parameter(self, param: str) -> bool:
        return param in SUPPORTED_PARAMS

    def supports_batch(self) -> bool:
        return True

    def translate_error(self, error: Exception) -> dict[str, Any]:
        if isinstance(error, AnyModelError):
            return {"code": error.code, "message": str(error), "metadata": error.metadata}
        status = getattr(error, "status", None) or getattr(error, "code", None) or 500
        return {
            "code": self._map_error_code(int(status)),
            "message": str(error),
            "metadata": {"provider_name": "openai", "raw": error},
        }


def create_openai_adapter(api_key: str, base_url: str | None = None) -> OpenAIAdapter:
    """Create an OpenAI provider adapter."""
    return OpenAIAdapter(api_key, base_url)
