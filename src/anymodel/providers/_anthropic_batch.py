"""Anthropic batch adapter."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from anymodel._types import AnyModelError
from anymodel.providers._adapter import NativeBatchStatus
from anymodel.utils._id import generate_id
from anymodel.utils._timeout import get_default_timeout
from anymodel.utils._token_estimate import resolve_max_tokens

ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096


def _map_stop_reason(reason: str) -> str:
    """Map Anthropic stop_reason to OpenAI finish_reason."""
    return {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "stop_sequence": "stop",
    }.get(reason, "stop")


def _translate_anthropic_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert an Anthropic message response to OpenAI ChatCompletion format."""
    content = ""
    tool_calls: list[dict[str, Any]] = []

    for block in msg.get("content", []):
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

    usage = msg.get("usage", {})
    prompt_tokens = usage.get("input_tokens", 0)
    completion_tokens = usage.get("output_tokens", 0)

    return {
        "id": generate_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"anthropic/{msg.get('model', 'unknown')}",
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": _map_stop_reason(msg.get("stop_reason", "end_turn")),
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _translate_to_anthropic_params(
    model: str, req: dict[str, Any],
) -> dict[str, Any]:
    """Convert an OpenAI-style batch request item to Anthropic params."""
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": resolve_max_tokens(
            model,
            req.get("messages", []),
            req.get("max_tokens") or DEFAULT_MAX_TOKENS,
        ),
    }

    messages = req.get("messages", [])
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system_msgs = [m for m in messages if m.get("role") != "system"]

    # Extract system prompt
    if system_msgs:
        params["system"] = "\n".join(
            m.get("content", "") if isinstance(m.get("content"), str) else ""
            for m in system_msgs
        )

    # Map messages
    params["messages"] = [
        {
            "role": "user" if m.get("role") == "tool" else m.get("role", "user"),
            "content": (
                [{"type": "tool_result", "tool_use_id": m["tool_call_id"],
                  "content": m.get("content", "") if isinstance(m.get("content"), str) else ""}]
                if m.get("tool_call_id")
                else m.get("content", "")
            ),
        }
        for m in non_system_msgs
    ]

    # Optional params
    if "temperature" in req:
        params["temperature"] = req["temperature"]
    if "top_p" in req:
        params["top_p"] = req["top_p"]
    if "top_k" in req:
        params["top_k"] = req["top_k"]
    if "stop" in req:
        stop = req["stop"]
        params["stop_sequences"] = stop if isinstance(stop, list) else [stop]

    # Map tools
    tools = req.get("tools")
    if tools:
        params["tools"] = [
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}}),
            }
            for t in tools
        ]

        tool_choice = req.get("tool_choice")
        if tool_choice:
            if tool_choice == "auto":
                params["tool_choice"] = {"type": "auto"}
            elif tool_choice == "required":
                params["tool_choice"] = {"type": "any"}
            elif tool_choice == "none":
                del params["tools"]
            elif isinstance(tool_choice, dict):
                params["tool_choice"] = {
                    "type": "tool",
                    "name": tool_choice["function"]["name"],
                }

    # Handle response_format
    response_format = req.get("response_format")
    if response_format:
        fmt_type = response_format.get("type")
        if fmt_type in ("json_object", "json_schema"):
            json_instruction = "Respond with valid JSON only. Do not include any text outside the JSON object."
            existing_system = params.get("system", "")
            params["system"] = (
                f"{json_instruction}\n\n{existing_system}" if existing_system
                else json_instruction
            )

    return params


class AnthropicBatchAdapter:
    """Anthropic native batch adapter."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=ANTHROPIC_API_BASE,
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": ANTHROPIC_VERSION,
                    "Content-Type": "application/json",
                },
                timeout=get_default_timeout(),
            )
        return self._client

    async def _api_request(
        self,
        path: str,
        *,
        method: str = "GET",
        body: Any = None,
    ) -> httpx.Response:
        client = self._get_client()
        if method == "GET":
            res = await client.get(path)
        else:
            if body is not None:
                res = await client.request(method, path, json=body)
            else:
                res = await client.request(method, path)

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
                502 if res.status_code >= 500 else res.status_code,
                msg or "Anthropic batch API error",
                {"provider_name": "anthropic", "raw": error_body},
            )

        return res

    async def create_batch(
        self,
        model: str,
        requests: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a batch to Anthropic's Messages Batches API."""
        batch_requests = [
            {
                "custom_id": req.get("custom_id", f"request-{i}"),
                "params": _translate_to_anthropic_params(model, req),
            }
            for i, req in enumerate(requests)
        ]

        res = await self._api_request(
            "/messages/batches",
            method="POST",
            body={"requests": batch_requests},
        )
        data = res.json()

        return {
            "providerBatchId": data["id"],
            "metadata": {
                "anthropic_type": data.get("type"),
                "created_at": data.get("created_at"),
            },
        }

    async def poll_batch(self, provider_batch_id: str) -> NativeBatchStatus:
        """Poll batch status."""
        res = await self._api_request(f"/messages/batches/{provider_batch_id}")
        data = res.json()

        counts = data.get("request_counts", {})
        total = (
            counts.get("processing", 0)
            + counts.get("succeeded", 0)
            + counts.get("errored", 0)
            + counts.get("canceled", 0)
            + counts.get("expired", 0)
        )

        if data.get("processing_status") == "ended":
            if counts.get("succeeded", 0) == 0 and (
                counts.get("errored", 0) > 0
                or counts.get("expired", 0) > 0
                or counts.get("canceled", 0) > 0
            ):
                status = "failed"
            elif data.get("cancel_initiated_at"):
                status = "cancelled"
            else:
                status = "completed"
        else:
            status = "processing"

        return NativeBatchStatus(
            status=status,
            total=total,
            completed=counts.get("succeeded", 0),
            failed=(
                counts.get("errored", 0)
                + counts.get("expired", 0)
                + counts.get("canceled", 0)
            ),
        )

    async def get_batch_results(self, provider_batch_id: str) -> list[dict[str, Any]]:
        """Download batch results as JSONL."""
        res = await self._api_request(f"/messages/batches/{provider_batch_id}/results")
        text = res.text

        results: list[dict[str, Any]] = []
        for line in text.strip().split("\n"):
            if not line:
                continue
            item = json.loads(line)

            result = item.get("result", {})
            if result.get("type") == "succeeded":
                results.append({
                    "custom_id": item["custom_id"],
                    "status": "success",
                    "response": _translate_anthropic_message(result["message"]),
                    "error": None,
                })
            else:
                error_type = result.get("type", "unknown")
                error_msg = (
                    result.get("error", {}).get("message")
                    or f"Batch item {error_type}"
                )
                results.append({
                    "custom_id": item["custom_id"],
                    "status": "error",
                    "response": None,
                    "error": {
                        "code": 408 if error_type == "expired" else 500,
                        "message": error_msg,
                    },
                })

        return results

    async def cancel_batch(self, provider_batch_id: str) -> None:
        """Cancel a batch."""
        await self._api_request(
            f"/messages/batches/{provider_batch_id}/cancel", method="POST",
        )


def create_anthropic_batch_adapter(api_key: str) -> AnthropicBatchAdapter:
    """Create an Anthropic batch adapter."""
    return AnthropicBatchAdapter(api_key)
