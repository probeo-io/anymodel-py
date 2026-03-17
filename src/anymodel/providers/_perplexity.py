"""Perplexity provider adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from anymodel._types import AnyModelError
from anymodel.utils._timeout import get_default_timeout

PERPLEXITY_API_BASE = "https://api.perplexity.ai"

SUPPORTED_PARAMS = frozenset({
    "temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty",
    "stream", "stop", "response_format", "tools", "tool_choice",
})

# Static model list — Perplexity has no /models endpoint
_MODELS = [
    {"id": "sonar", "name": "Sonar", "context": 128000, "max_output": 4096},
    {"id": "sonar-pro", "name": "Sonar Pro", "context": 200000, "max_output": 8192},
    {"id": "sonar-reasoning", "name": "Sonar Reasoning", "context": 128000, "max_output": 8192},
    {"id": "sonar-reasoning-pro", "name": "Sonar Reasoning Pro", "context": 128000, "max_output": 16384},
    {"id": "sonar-deep-research", "name": "Sonar Deep Research", "context": 128000, "max_output": 16384},
    {"id": "r1-1776", "name": "R1 1776", "context": 128000, "max_output": 16384},
]


class PerplexityAdapter:
    """Perplexity chat completion adapter with citation support."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._base_url = PERPLEXITY_API_BASE
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "perplexity"

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

    async def send_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a non-streaming chat completion request."""
        body = self._build_request_body(request)
        client = self._get_client()
        res = await client.post("/chat/completions", json=body)

        if res.status_code >= 400:
            try:
                error_body = res.json()
            except Exception:
                error_body = {"message": res.reason_phrase}
            msg = error_body.get("error", {}).get("message") or error_body.get("message") or res.reason_phrase
            raise AnyModelError(self._map_error_code(res.status_code), msg or "Unknown Perplexity error", {"provider_name": "perplexity", "raw": error_body})

        data = res.json()
        result = self._translate_response(data)

        # Preserve citations if present
        if "citations" in data:
            result["citations"] = data["citations"]

        return result

    async def send_streaming_request(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Send a streaming chat completion request."""
        body = self._build_request_body({**request, "stream": True})
        client = self._get_client()
        req = client.build_request("POST", "/chat/completions", json=body)
        res = await client.send(req, stream=True)

        if res.status_code >= 400:
            error_text = await res.aread()
            await res.aclose()
            try:
                error_body = json.loads(error_text)
            except Exception:
                error_body = {"message": res.reason_phrase}
            msg = error_body.get("error", {}).get("message") or res.reason_phrase
            raise AnyModelError(self._map_error_code(res.status_code), msg or "Unknown Perplexity error", {"provider_name": "perplexity", "raw": error_body})

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
                    chunk["model"] = f"perplexity/{chunk.get('model', '')}"
                    yield chunk
        finally:
            await res.aclose()

    def _translate_response(self, data: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": self._re_prefix_id(data.get("id", "")),
            "object": "chat.completion",
            "created": data.get("created", 0),
            "model": f"perplexity/{data.get('model', '')}",
            "choices": data.get("choices", []),
            "usage": data.get("usage", {}),
        }

    async def list_models(self) -> list[dict[str, Any]]:
        """Return available Perplexity models (static list)."""
        return [
            {
                "id": f"perplexity/{m['id']}",
                "name": m["name"],
                "created": 0,
                "description": "",
                "context_length": m["context"],
                "pricing": {"prompt": "0", "completion": "0"},
                "architecture": {
                    "modality": "text->text",
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                    "tokenizer": "unknown",
                },
                "top_provider": {
                    "context_length": m["context"],
                    "max_completion_tokens": m["max_output"],
                    "is_moderated": False,
                },
                "supported_parameters": list(SUPPORTED_PARAMS),
            }
            for m in _MODELS
        ]

    def supports_parameter(self, param: str) -> bool:
        return param in SUPPORTED_PARAMS

    def supports_batch(self) -> bool:
        return False

    def translate_error(self, error: Exception) -> dict[str, Any]:
        if isinstance(error, AnyModelError):
            return {"code": error.code, "message": str(error), "metadata": error.metadata}
        status = getattr(error, "status", None) or getattr(error, "code", None) or 500
        return {
            "code": self._map_error_code(int(status)),
            "message": str(error),
            "metadata": {"provider_name": "perplexity", "raw": error},
        }


def create_perplexity_adapter(api_key: str) -> PerplexityAdapter:
    """Create a Perplexity provider adapter."""
    return PerplexityAdapter(api_key)
