"""xAI provider adapter — native chat + Responses (web search) support."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from anymodel._types import AnyModelError
from anymodel.utils._timeout import get_default_timeout

XAI_API_BASE = "https://api.x.ai/v1"

SUPPORTED_PARAMS = frozenset({
    "temperature", "max_tokens", "top_p", "seed", "stop", "stream",
    "response_format", "tools", "tool_choice", "cache",
})

_MODELS = [
    {"id": "grok-4.5", "name": "Grok 4.5", "context": 256000, "max_output": 128000},
    {"id": "grok-4.3", "name": "Grok 4.3", "context": 256000, "max_output": 128000},
    {"id": "grok-4.20", "name": "Grok 4.20", "context": 256000, "max_output": 128000},
]


def _map_error_code(status: int) -> int:
    if status in (401, 403):
        return 401
    if status == 429:
        return 429
    if status in (400, 422):
        return 400
    if status >= 500:
        return 502
    return status


def _re_prefix_id(id_: str | None) -> str:
    value = id_ or f"xai-{int(time.time() * 1000)}"
    return value if value.startswith("gen-") else f"gen-{value}"


def _has_web_search_tool(tools: list[dict[str, Any]] | None) -> bool:
    return any(isinstance(t, dict) and t.get("type") == "web_search" for t in tools or [])


def _responses_input(request: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"role": m.get("role"), "content": m.get("content")} for m in request["messages"]]


def _text_from_responses(data: dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str):
        return output_text
    chunks: list[str] = []
    for item in data.get("output") or []:
        if item.get("type") != "message" or not isinstance(item.get("content"), list):
            continue
        for part in item["content"]:
            text = part.get("text") if isinstance(part, dict) else None
            if text is None and isinstance(part, dict):
                text = part.get("content")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks).strip()


def _usage_from_responses(data: dict[str, Any]) -> dict[str, Any]:
    usage = data.get("usage") or {}
    prompt = usage.get("input_tokens", 0) or 0
    completion = usage.get("output_tokens", 0) or 0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": usage.get("total_tokens", prompt + completion) or 0,
    }


class XAIAdapter:
    """xAI (Grok) chat completion adapter."""

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = (base_url or XAI_API_BASE).rstrip("/")
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "xai"

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

    async def _make_request(self, path: str, body: dict[str, Any] | None = None) -> httpx.Response:
        client = self._get_client()
        res = await client.post(path, json=body)
        if res.status_code >= 400:
            try:
                error_body = res.json()
            except Exception:
                error_body = {"message": res.reason_phrase}
            msg = error_body.get("error", {}).get("message") or error_body.get("message") or res.reason_phrase
            raise AnyModelError(
                _map_error_code(res.status_code),
                msg or "Unknown xAI error",
                {"provider_name": "xai", "raw": error_body},
            )
        return res

    def _build_chat_body(self, request: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {"model": request["model"], "messages": request["messages"]}
        for param in ("temperature", "max_tokens", "top_p", "seed", "stop",
                      "stream", "response_format", "tools", "tool_choice"):
            if param in request:
                body[param] = request[param]
        cache = request.get("cache")
        if cache and cache.get("key") is not None:
            body["prompt_cache_key"] = cache["key"]
        return body

    def _build_responses_body(self, request: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request["model"],
            "input": _responses_input(request),
            "tools": request.get("tools"),
        }
        for param in ("temperature", "top_p", "stop"):
            if param in request:
                body[param] = request[param]
        cache = request.get("cache")
        if cache and cache.get("key") is not None:
            body["prompt_cache_key"] = cache["key"]
        if request.get("response_format", {}).get("type") == "json_object":
            body["text"] = {"format": {"type": "json_object"}}
        return body

    def _translate_chat_response(self, data: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": _re_prefix_id(data.get("id")),
            "object": "chat.completion",
            "created": data.get("created", int(time.time())),
            "model": f"xai/{data.get('model', '')}",
            "choices": data.get("choices", []),
            "usage": data.get("usage", {}),
        }
        if data.get("citations"):
            result["citations"] = data["citations"]
        return result

    def _translate_responses_response(self, data: dict[str, Any], model: str | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": _re_prefix_id(data.get("id")),
            "object": "chat.completion",
            "created": data.get("created_at", int(time.time())),
            "model": f"xai/{data.get('model') or model or 'unknown'}",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": _text_from_responses(data)},
                "finish_reason": "stop",
            }],
            "usage": _usage_from_responses(data),
        }
        if data.get("citations"):
            result["citations"] = data["citations"]
        if data.get("output") is not None:
            result["output"] = data["output"]
        if data.get("server_side_tool_usage"):
            result["server_side_tool_usage"] = data["server_side_tool_usage"]
        usage = data.get("usage")
        if usage:
            result["raw_usage"] = usage
            if usage.get("server_side_tool_usage_details"):
                result["server_side_tool_usage_details"] = usage["server_side_tool_usage_details"]
        return result

    async def send_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a non-streaming chat completion request."""
        if _has_web_search_tool(request.get("tools")):
            body = self._build_responses_body(request)
            res = await self._make_request("/responses", body)
            return self._translate_responses_response(res.json(), request.get("model"))
        body = self._build_chat_body(request)
        res = await self._make_request("/chat/completions", body)
        return self._translate_chat_response(res.json())

    async def send_request_with_meta(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a request and return completion + response headers."""
        completion = await self.send_request(request)
        return {"completion": completion, "meta": {"headers": {}}}

    async def send_streaming_request(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """xAI streaming is not implemented in the native adapter yet."""
        raise AnyModelError(400, "xAI streaming is not implemented in the native adapter yet", {"provider_name": "xai"})

    async def list_models(self) -> list[dict[str, Any]]:
        """Return available xAI (Grok) models (static list)."""
        return [
            {
                "id": f"xai/{m['id']}",
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
            "code": _map_error_code(int(status)),
            "message": str(error),
            "metadata": {"provider_name": "xai", "raw": error},
        }


def create_xai_adapter(api_key: str, base_url: str | None = None) -> XAIAdapter:
    """Create an xAI provider adapter."""
    return XAIAdapter(api_key, base_url)
