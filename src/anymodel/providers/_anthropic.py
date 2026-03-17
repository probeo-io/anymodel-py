"""Anthropic provider adapter."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from anymodel._types import AnyModelError
from anymodel.utils._id import generate_id
from anymodel.utils._timeout import get_default_timeout

ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096

SUPPORTED_PARAMS = frozenset({
    "temperature", "max_tokens", "top_p", "top_k", "stop", "stream",
    "tools", "tool_choice", "response_format",
})

FALLBACK_MODELS: list[dict[str, Any]] = [
    {"id": "anthropic/claude-opus-4-6", "name": "Claude Opus 4.6", "created": 0, "description": "Most capable model", "context_length": 200000, "pricing": {"prompt": "0.000015", "completion": "0.000075"}, "architecture": {"modality": "text+image->text", "input_modalities": ["text", "image"], "output_modalities": ["text"], "tokenizer": "claude"}, "top_provider": {"context_length": 200000, "max_completion_tokens": 32768, "is_moderated": False}, "supported_parameters": list(SUPPORTED_PARAMS)},
    {"id": "anthropic/claude-sonnet-4-6", "name": "Claude Sonnet 4.6", "created": 0, "description": "Best balance of speed and capability", "context_length": 200000, "pricing": {"prompt": "0.000003", "completion": "0.000015"}, "architecture": {"modality": "text+image->text", "input_modalities": ["text", "image"], "output_modalities": ["text"], "tokenizer": "claude"}, "top_provider": {"context_length": 200000, "max_completion_tokens": 16384, "is_moderated": False}, "supported_parameters": list(SUPPORTED_PARAMS)},
    {"id": "anthropic/claude-haiku-4-5", "name": "Claude Haiku 4.5", "created": 0, "description": "Fast and compact", "context_length": 200000, "pricing": {"prompt": "0.000001", "completion": "0.000005"}, "architecture": {"modality": "text+image->text", "input_modalities": ["text", "image"], "output_modalities": ["text"], "tokenizer": "claude"}, "top_provider": {"context_length": 200000, "max_completion_tokens": 8192, "is_moderated": False}, "supported_parameters": list(SUPPORTED_PARAMS)},
]


def _map_error_code(status: int) -> int:
    if status in (401, 403):
        return 401
    if status == 429:
        return 429
    if status in (400, 422):
        return 400
    if status >= 500 or status == 529:
        return 502
    return status


def _map_stop_reason(reason: str) -> str:
    return {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "stop_sequence": "stop",
    }.get(reason, "stop")


class AnthropicAdapter:
    """Anthropic chat completion adapter."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "anthropic"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=ANTHROPIC_API_BASE,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self._api_key,
                    "anthropic-version": ANTHROPIC_VERSION,
                },
                timeout=get_default_timeout(),
            )
        return self._client

    def _translate_request(self, request: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request["model"],
            "max_tokens": request.get("max_tokens", DEFAULT_MAX_TOKENS),
        }

        # Extract system messages
        system_msgs = [m for m in request["messages"] if m.get("role") == "system"]
        non_system_msgs = [m for m in request["messages"] if m.get("role") != "system"]

        if system_msgs:
            body["system"] = "\n".join(
                m.get("content", "") if isinstance(m.get("content"), str) else ""
                for m in system_msgs
            )

        # Map messages
        body["messages"] = []
        for m in non_system_msgs:
            role = "user" if m.get("role") == "tool" else m.get("role", "user")
            if m.get("tool_call_id"):
                content = [{"type": "tool_result", "tool_use_id": m["tool_call_id"],
                            "content": m.get("content", "") if isinstance(m.get("content"), str) else ""}]
            else:
                content = m.get("content", "")
            body["messages"].append({"role": role, "content": content})

        # Optional params
        if "temperature" in request:
            body["temperature"] = request["temperature"]
        if "top_p" in request:
            body["top_p"] = request["top_p"]
        if "top_k" in request:
            body["top_k"] = request["top_k"]
        if "stop" in request:
            stop = request["stop"]
            body["stop_sequences"] = stop if isinstance(stop, list) else [stop]
        if request.get("stream"):
            body["stream"] = True

        # Map tools
        tools = request.get("tools")
        if tools:
            body["tools"] = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}}),
                }
                for t in tools
            ]

            tool_choice = request.get("tool_choice")
            if tool_choice == "auto":
                body["tool_choice"] = {"type": "auto"}
            elif tool_choice == "required":
                body["tool_choice"] = {"type": "any"}
            elif tool_choice == "none":
                del body["tools"]
            elif isinstance(tool_choice, dict):
                body["tool_choice"] = {"type": "tool", "name": tool_choice["function"]["name"]}

        # Handle response_format
        response_format = request.get("response_format")
        if response_format and response_format.get("type") in ("json_object", "json_schema"):
            json_instruction = "Respond with valid JSON only. Do not include any text outside the JSON object."
            body["system"] = f"{json_instruction}\n\n{body['system']}" if body.get("system") else json_instruction

        return body

    def _translate_response(self, data: dict[str, Any]) -> dict[str, Any]:
        content = ""
        tool_calls = []

        for block in data.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })

        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls

        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        return {
            "id": generate_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": f"anthropic/{data.get('model', '')}",
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": _map_stop_reason(data.get("stop_reason", "end_turn")),
            }],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }

    async def send_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a non-streaming chat completion request."""
        body = self._translate_request(request)
        client = self._get_client()
        res = await client.post("/messages", json=body)

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
                _map_error_code(res.status_code),
                msg or "Unknown Anthropic error",
                {"provider_name": "anthropic", "raw": error_body},
            )

        return self._translate_response(res.json())

    async def send_streaming_request(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Send a streaming chat completion request."""
        body = self._translate_request({**request, "stream": True})
        client = self._get_client()
        req = client.build_request("POST", "/messages", json=body)
        res = await client.send(req, stream=True)

        if res.status_code >= 400:
            error_text = await res.aread()
            await res.aclose()
            try:
                error_body = json.loads(error_text)
            except Exception:
                error_body = {"message": res.reason_phrase}
            msg = error_body.get("error", {}).get("message") or res.reason_phrase
            raise AnyModelError(
                _map_error_code(res.status_code),
                msg or "Unknown Anthropic error",
                {"provider_name": "anthropic", "raw": error_body},
            )

        return self._iter_sse(res)

    async def _iter_sse(self, res: httpx.Response) -> AsyncIterator[dict[str, Any]]:
        gen_id = generate_id()
        created = int(time.time())
        model = ""
        input_usage: dict[str, Any] = {}

        try:
            async for line in res.aiter_lines():
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data: "):
                    continue

                data = json.loads(line[6:])
                event_type = data.get("type", "")

                if event_type == "message_start":
                    msg = data.get("message", {})
                    model = f"anthropic/{msg.get('model', '')}"
                    input_usage = msg.get("usage", {})
                    yield {
                        "id": gen_id, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    }

                elif event_type == "content_block_start":
                    block = data.get("content_block", {})
                    if block.get("type") == "tool_use":
                        yield {
                            "id": gen_id, "object": "chat.completion.chunk", "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {"tool_calls": [{
                                "id": block["id"], "type": "function",
                                "function": {"name": block["name"], "arguments": ""},
                            }]}, "finish_reason": None}],
                        }

                elif event_type == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield {
                            "id": gen_id, "object": "chat.completion.chunk", "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {"content": delta["text"]}, "finish_reason": None}],
                        }
                    elif delta.get("type") == "input_json_delta":
                        yield {
                            "id": gen_id, "object": "chat.completion.chunk", "created": created, "model": model,
                            "choices": [{"index": 0, "delta": {"tool_calls": [{
                                "id": "", "type": "function",
                                "function": {"name": "", "arguments": delta.get("partial_json", "")},
                            }]}, "finish_reason": None}],
                        }

                elif event_type == "message_delta":
                    msg_delta = data.get("delta", {})
                    output_tokens = data.get("usage", {}).get("output_tokens", 0)
                    input_tokens = input_usage.get("input_tokens", 0)
                    yield {
                        "id": gen_id, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {},
                                     "finish_reason": _map_stop_reason(msg_delta.get("stop_reason", "end_turn"))}],
                        "usage": {
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                        },
                    }
        finally:
            await res.aclose()

    async def list_models(self) -> list[dict[str, Any]]:
        """Fetch available models from Anthropic."""
        try:
            client = self._get_client()
            res = await client.get("/models")
            if res.status_code >= 400:
                return FALLBACK_MODELS
            data = res.json()
            return [
                {
                    "id": f"anthropic/{m['id']}",
                    "name": m.get("display_name", m["id"]),
                    "created": 0,
                    "description": m.get("display_name", ""),
                    "context_length": 200000,
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"modality": "text+image->text", "input_modalities": ["text", "image"],
                                     "output_modalities": ["text"], "tokenizer": "claude"},
                    "top_provider": {"context_length": 200000, "max_completion_tokens": 16384, "is_moderated": False},
                    "supported_parameters": list(SUPPORTED_PARAMS),
                }
                for m in data.get("data", [])
                if m.get("type") == "model"
            ]
        except Exception:
            return FALLBACK_MODELS

    def supports_parameter(self, param: str) -> bool:
        return param in SUPPORTED_PARAMS

    def supports_batch(self) -> bool:
        return True

    def translate_error(self, error: Exception) -> dict[str, Any]:
        if isinstance(error, AnyModelError):
            return {"code": error.code, "message": str(error), "metadata": error.metadata}
        status = getattr(error, "status", None) or getattr(error, "code", None) or 500
        return {
            "code": _map_error_code(int(status)),
            "message": str(error),
            "metadata": {"provider_name": "anthropic", "raw": error},
        }


def create_anthropic_adapter(api_key: str) -> AnthropicAdapter:
    """Create an Anthropic provider adapter."""
    return AnthropicAdapter(api_key)
