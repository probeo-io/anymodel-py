"""OpenAI provider adapter."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx

from anymodel._types import AnyModelError
from anymodel.utils._timeout import get_default_timeout, get_flex_timeout

_MAX_COMPLETION_TOKENS_RE = re.compile(r"^(o[1-9]|gpt-5|gpt-4o)")

_RATE_LIMIT_HEADERS = (
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
    "retry-after",
)


def _uses_max_completion_tokens(model: str) -> bool:
    """Return True if *model* should use ``max_completion_tokens`` instead of ``max_tokens``."""
    return bool(_MAX_COMPLETION_TOKENS_RE.search(model))


def _extract_rate_limit_headers(response: httpx.Response) -> dict[str, str]:
    """Extract rate-limit headers from an httpx response."""
    headers: dict[str, str] = {}
    for name in _RATE_LIMIT_HEADERS:
        value = response.headers.get(name)
        if value is not None:
            headers[name] = value
    return headers

OPENAI_API_BASE = "https://api.openai.com/v1"

SUPPORTED_PARAMS = frozenset({
    "temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty",
    "seed", "stop", "stream", "logprobs", "top_logprobs", "response_format",
    "tools", "tool_choice", "user", "logit_bias", "service_tier", "cache", "reasoning",
})

# Params handled specially (not copied verbatim into the request body)
_SPECIAL_PARAMS = frozenset({"cache", "reasoning"})


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
        for param in SUPPORTED_PARAMS - _SPECIAL_PARAMS:
            if param in request:
                body[param] = request[param]

        # Translate max_tokens → max_completion_tokens for newer models
        if "max_tokens" in body and _uses_max_completion_tokens(request["model"]):
            body["max_completion_tokens"] = body.pop("max_tokens")

        self._apply_cache(body, request.get("cache"))
        return body

    def _apply_cache(self, body: dict[str, Any], cache: dict[str, Any] | None) -> None:
        """Apply prompt cache key/retention onto a request body, in place."""
        if not cache:
            return
        if cache.get("key") is not None:
            body["prompt_cache_key"] = cache["key"]
        if cache.get("ttl") is not None:
            body["prompt_cache_retention"] = "24h" if cache["ttl"] == "24h" else "in_memory"

    # ─── Responses API (reasoning models, web search) ──────────────────────

    def _uses_responses(self, request: dict[str, Any]) -> bool:
        return self._has_web_search_tool(request.get("tools")) or request.get("reasoning") is not None

    @staticmethod
    def _has_web_search_tool(tools: list[dict[str, Any]] | None) -> bool:
        return any(isinstance(t, dict) and t.get("type") == "web_search" for t in tools or [])

    def _responses_tools(self, tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        result = []
        for t in tools:
            if t.get("type") == "function":
                fn = t.get("function", {})
                result.append({
                    "type": "function",
                    "name": fn.get("name"),
                    "description": fn.get("description"),
                    "parameters": fn.get("parameters"),
                })
            else:
                result.append(t)
        return result

    def _responses_tool_choice(self, choice: Any) -> Any:
        if choice is None or isinstance(choice, str):
            return choice
        fn = choice.get("function", {})
        return {"type": "function", "name": fn.get("name")}

    def _responses_input(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        input_items: list[dict[str, Any]] = []
        for m in request["messages"]:
            role = m.get("role")
            if role == "tool":
                input_items.append({
                    "type": "function_call_output",
                    "call_id": m.get("tool_call_id"),
                    "output": m.get("content"),
                })
            elif role == "assistant" and m.get("tool_calls"):
                for call in m["tool_calls"]:
                    fn = call.get("function", {})
                    input_items.append({
                        "type": "function_call",
                        "call_id": call.get("id"),
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments"),
                    })
            else:
                input_items.append({"role": role, "content": m.get("content")})
        return input_items

    def _build_responses_body(self, request: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request["model"],
            "input": self._responses_input(request),
            "tools": self._responses_tools(request.get("tools")),
            "tool_choice": self._responses_tool_choice(request.get("tool_choice")) or "required",
            "include": ["web_search_call.action.sources"],
        }
        if "max_tokens" in request:
            body["max_output_tokens"] = request["max_tokens"]
        for param in ("temperature", "top_p", "seed", "user", "service_tier"):
            if param in request:
                body[param] = request[param]
        if request.get("reasoning") is not None:
            body["reasoning"] = request["reasoning"]
        self._apply_cache(body, request.get("cache"))
        if request.get("response_format", {}).get("type") == "json_object":
            body["text"] = {"format": {"type": "json_object"}}
        return body

    def _text_from_responses(self, data: dict[str, Any]) -> str:
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

    def _usage_from_responses(self, data: dict[str, Any]) -> dict[str, Any]:
        usage = data.get("usage") or {}
        prompt = usage.get("input_tokens", 0) or 0
        completion = usage.get("output_tokens", 0) or 0
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": usage.get("total_tokens", prompt + completion) or 0,
        }

    def _translate_responses_response(self, data: dict[str, Any]) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": self._text_from_responses(data)}
        tool_calls = [
            {
                "id": item.get("call_id"),
                "type": "function",
                "function": {"name": item.get("name"), "arguments": item.get("arguments")},
            }
            for item in data.get("output") or []
            if item.get("type") == "function_call"
        ]
        finish_reason = "tool_calls" if tool_calls else "stop"
        if tool_calls:
            message["tool_calls"] = tool_calls

        result: dict[str, Any] = {
            "id": self._re_prefix_id(data.get("id", "")),
            "object": "chat.completion",
            "created": data.get("created_at", 0),
            "model": f"openai/{data.get('model', '')}",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": self._usage_from_responses(data),
        }
        if data.get("output") is not None:
            result["output"] = data["output"]
        usage = data.get("usage")
        if usage:
            result["raw_usage"] = usage
            if usage.get("server_side_tool_usage_details"):
                result["server_side_tool_usage_details"] = usage["server_side_tool_usage_details"]
        return result

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
        timeout = self._request_timeout(request)
        if self._uses_responses(request):
            body = self._build_responses_body(request)
            res = await self._make_request("/responses", body, timeout=timeout)
            return self._translate_responses_response(res.json())
        body = self._build_request_body(request)
        res = await self._make_request("/chat/completions", body, timeout=timeout)
        data = res.json()
        return self._translate_response(data)

    async def send_request_with_meta(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a request and return completion + response headers."""
        timeout = self._request_timeout(request)
        if self._uses_responses(request):
            body = self._build_responses_body(request)
            res = await self._make_request("/responses", body, timeout=timeout)
            return {
                "completion": self._translate_responses_response(res.json()),
                "meta": {"headers": _extract_rate_limit_headers(res)},
            }
        body = self._build_request_body(request)
        res = await self._make_request("/chat/completions", body, timeout=timeout)
        data = res.json()
        return {
            "completion": self._translate_response(data),
            "meta": {"headers": _extract_rate_limit_headers(res)},
        }

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
