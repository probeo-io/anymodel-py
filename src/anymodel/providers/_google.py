"""Google Gemini provider adapter."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from anymodel._types import AnyModelError
from anymodel.utils._id import generate_id

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

SUPPORTED_PARAMS = frozenset({
    "temperature", "max_tokens", "top_p", "top_k", "stop", "stream",
    "tools", "tool_choice", "response_format",
})

FALLBACK_MODELS: list[dict[str, Any]] = [
    {"id": "google/gemini-2.5-pro", "name": "Gemini 2.5 Pro", "created": 0, "description": "Most capable Gemini", "context_length": 1000000, "pricing": {"prompt": "0", "completion": "0"}, "architecture": {"modality": "text+image->text", "input_modalities": ["text", "image"], "output_modalities": ["text"], "tokenizer": "gemini"}, "top_provider": {"context_length": 1000000, "max_completion_tokens": 65536, "is_moderated": True}, "supported_parameters": list(SUPPORTED_PARAMS)},
    {"id": "google/gemini-2.5-flash", "name": "Gemini 2.5 Flash", "created": 0, "description": "Fast Gemini", "context_length": 1000000, "pricing": {"prompt": "0", "completion": "0"}, "architecture": {"modality": "text+image->text", "input_modalities": ["text", "image"], "output_modalities": ["text"], "tokenizer": "gemini"}, "top_provider": {"context_length": 1000000, "max_completion_tokens": 65536, "is_moderated": True}, "supported_parameters": list(SUPPORTED_PARAMS)},
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


def _map_finish_reason(reason: str) -> str:
    return {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
    }.get(reason, "stop")


class GoogleAdapter:
    """Google Gemini chat completion adapter."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "google"

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=GEMINI_API_BASE,
                headers={"Content-Type": "application/json"},
                timeout=120.0,
            )
        return self._client

    def _translate_request(self, request: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {}

        # Extract system messages
        system_msgs = [m for m in request["messages"] if m.get("role") == "system"]
        non_system_msgs = [m for m in request["messages"] if m.get("role") != "system"]

        if system_msgs:
            system_text = "\n".join(
                m.get("content", "") if isinstance(m.get("content"), str) else ""
                for m in system_msgs
            )
            body["systemInstruction"] = {"parts": [{"text": system_text}]}

        # Map messages to contents
        contents = []
        for m in non_system_msgs:
            role = m.get("role", "user")
            gemini_role = "model" if role == "assistant" else "user"
            content = m.get("content", "")

            if m.get("tool_call_id"):
                contents.append({
                    "role": "function",
                    "parts": [{"functionResponse": {
                        "name": m.get("name", ""),
                        "response": {"result": content if isinstance(content, str) else ""},
                    }}],
                })
            elif m.get("tool_calls"):
                parts = []
                for tc in m["tool_calls"]:
                    args = tc["function"].get("arguments", "{}")
                    parts.append({"functionCall": {
                        "name": tc["function"]["name"],
                        "args": json.loads(args) if isinstance(args, str) else args,
                    }})
                contents.append({"role": "model", "parts": parts})
            else:
                if isinstance(content, str):
                    contents.append({"role": gemini_role, "parts": [{"text": content}]})
                elif isinstance(content, list):
                    parts = []
                    for part in content:
                        if part.get("type") == "text":
                            parts.append({"text": part.get("text", "")})
                        elif part.get("type") == "image_url":
                            url = part.get("image_url", {}).get("url", "")
                            if url.startswith("data:"):
                                mime, _, data = url.partition(";base64,")
                                parts.append({"inlineData": {
                                    "mimeType": mime.replace("data:", ""),
                                    "data": data,
                                }})
                    contents.append({"role": gemini_role, "parts": parts})

        body["contents"] = contents

        # Generation config
        gen_config: dict[str, Any] = {}
        if "temperature" in request:
            gen_config["temperature"] = request["temperature"]
        if "max_tokens" in request:
            gen_config["maxOutputTokens"] = request["max_tokens"]
        if "top_p" in request:
            gen_config["topP"] = request["top_p"]
        if "top_k" in request:
            gen_config["topK"] = request["top_k"]
        if "stop" in request:
            stop = request["stop"]
            gen_config["stopSequences"] = stop if isinstance(stop, list) else [stop]

        response_format = request.get("response_format")
        if response_format:
            if response_format.get("type") == "json_object":
                gen_config["responseMimeType"] = "application/json"
            elif response_format.get("type") == "json_schema":
                gen_config["responseMimeType"] = "application/json"
                schema = response_format.get("json_schema", {}).get("schema")
                if schema:
                    gen_config["responseSchema"] = schema

        if gen_config:
            body["generationConfig"] = gen_config

        # Tools
        tools = request.get("tools")
        if tools:
            body["tools"] = [{"functionDeclarations": [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "parameters": t["function"].get("parameters", {"type": "object", "properties": {}}),
                }
                for t in tools
            ]}]

        return body

    def _translate_response(self, data: dict[str, Any], model: str) -> dict[str, Any]:
        candidates = data.get("candidates", [{}])
        candidate = candidates[0] if candidates else {}
        parts = candidate.get("content", {}).get("parts", [])

        content = ""
        tool_calls = []
        for part in parts:
            if "text" in part:
                content += part["text"]
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append({
                    "id": generate_id("call"),
                    "type": "function",
                    "function": {
                        "name": fc["name"],
                        "arguments": json.dumps(fc.get("args", {})),
                    },
                })

        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls

        usage_meta = data.get("usageMetadata", {})
        prompt_tokens = usage_meta.get("promptTokenCount", 0)
        completion_tokens = usage_meta.get("candidatesTokenCount", 0)

        return {
            "id": generate_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": f"google/{model}",
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": _map_finish_reason(candidate.get("finishReason", "STOP")),
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def send_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a non-streaming chat completion request."""
        model = request["model"]
        body = self._translate_request(request)
        client = self._get_client()
        url = f"/models/{model}:generateContent?key={self._api_key}"
        res = await client.post(url, json=body)

        if res.status_code >= 400:
            try:
                error_body = res.json()
            except Exception:
                error_body = {"message": res.reason_phrase}
            msg = error_body.get("error", {}).get("message") or res.reason_phrase
            raise AnyModelError(
                _map_error_code(res.status_code),
                msg or "Unknown Google error",
                {"provider_name": "google", "raw": error_body},
            )

        return self._translate_response(res.json(), model)

    async def send_streaming_request(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Send a streaming chat completion request."""
        model = request["model"]
        body = self._translate_request(request)
        client = self._get_client()
        url = f"/models/{model}:streamGenerateContent?key={self._api_key}&alt=sse"
        req = client.build_request("POST", url, json=body)
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
                msg or "Unknown Google error",
                {"provider_name": "google", "raw": error_body},
            )

        return self._iter_sse(res, model)

    async def _iter_sse(self, res: httpx.Response, model: str) -> AsyncIterator[dict[str, Any]]:
        gen_id = generate_id()
        created = int(time.time())
        full_model = f"google/{model}"
        first = True

        try:
            async for line in res.aiter_lines():
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data: "):
                    continue

                data = json.loads(line[6:])
                candidates = data.get("candidates", [{}])
                candidate = candidates[0] if candidates else {}
                parts = candidate.get("content", {}).get("parts", [])

                delta: dict[str, Any] = {}
                if first:
                    delta["role"] = "assistant"
                    first = False

                for part in parts:
                    if "text" in part:
                        delta["content"] = part["text"]

                finish_reason = None
                if candidate.get("finishReason"):
                    finish_reason = _map_finish_reason(candidate["finishReason"])

                chunk: dict[str, Any] = {
                    "id": gen_id, "object": "chat.completion.chunk", "created": created, "model": full_model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                }

                usage_meta = data.get("usageMetadata")
                if usage_meta:
                    prompt_tokens = usage_meta.get("promptTokenCount", 0)
                    completion_tokens = usage_meta.get("candidatesTokenCount", 0)
                    chunk["usage"] = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    }

                yield chunk
        finally:
            await res.aclose()

    async def list_models(self) -> list[dict[str, Any]]:
        """Fetch available models from Google."""
        try:
            client = self._get_client()
            res = await client.get(f"/models?key={self._api_key}")
            if res.status_code >= 400:
                return FALLBACK_MODELS

            data = res.json()
            models = []
            for m in data.get("models", []):
                model_id = m.get("name", "").replace("models/", "")
                if not model_id.startswith("gemini"):
                    continue
                methods = m.get("supportedGenerationMethods", [])
                if "generateContent" not in methods:
                    continue

                models.append({
                    "id": f"google/{model_id}",
                    "name": m.get("displayName", model_id),
                    "created": 0,
                    "description": m.get("description", ""),
                    "context_length": m.get("inputTokenLimit", 128000),
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"modality": "text+image->text", "input_modalities": ["text", "image"],
                                     "output_modalities": ["text"], "tokenizer": "gemini"},
                    "top_provider": {"context_length": m.get("inputTokenLimit", 128000),
                                     "max_completion_tokens": m.get("outputTokenLimit", 8192),
                                     "is_moderated": True},
                    "supported_parameters": list(SUPPORTED_PARAMS),
                })
            return models
        except Exception:
            return FALLBACK_MODELS

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
            "metadata": {"provider_name": "google", "raw": error},
        }


def create_google_adapter(api_key: str) -> GoogleAdapter:
    """Create a Google Gemini provider adapter."""
    return GoogleAdapter(api_key)
