"""xAI native batch adapter."""

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

XAI_API_BASE = "https://api.x.ai/v1"


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        part.get("text", "") for part in content if part.get("type") == "text" and part.get("text")
    )


def _content_to_responses_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if part.get("type") == "text":
            parts.append({"type": "text", "text": part.get("text", "")})
        elif part.get("type") == "image_url":
            parts.append({"type": "image_url", "image_url": part.get("image_url")})
    return parts if parts else ""


def _translate_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    translated = []
    for m in messages:
        if m.get("role") == "tool":
            translated.append({"role": "user", "content": _content_to_text(m.get("content"))})
            continue
        item: dict[str, Any] = {"role": m.get("role"), "content": _content_to_responses_content(m.get("content"))}
        if m.get("name"):
            item["name"] = m["name"]
        if m.get("tool_calls"):
            item["tool_calls"] = m["tool_calls"]
        if m.get("tool_call_id"):
            item["tool_call_id"] = m["tool_call_id"]
        translated.append(item)
    return translated


def _translate_request(model: str, req: dict[str, Any]) -> dict[str, Any]:
    responses: dict[str, Any] = {
        "model": model,
        "input": _translate_input(req.get("messages", [])),
        "max_output_tokens": req.get("max_tokens") or resolve_max_tokens(model, req.get("messages", [])),
    }
    for key in ("temperature", "top_p", "stop"):
        if key in req:
            responses[key] = req[key]
    cache = req.get("cache")
    if cache and cache.get("key") is not None:
        responses["prompt_cache_key"] = cache["key"]

    tools = req.get("tools")
    if tools:
        function_tools = [t for t in tools if t.get("type") == "function"]
        if function_tools:
            responses["tools"] = [
                {
                    "type": "function",
                    "name": t["function"]["name"],
                    "description": t["function"].get("description"),
                    "parameters": t["function"].get("parameters") or {"type": "object", "properties": {}},
                }
                for t in function_tools
            ]

    if "tool_choice" in req:
        responses["tool_choice"] = req["tool_choice"]

    response_format = req.get("response_format")
    if response_format:
        if response_format.get("type") == "json_schema":
            schema = response_format.get("json_schema", {})
            responses["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": schema.get("name"),
                    "schema": schema.get("schema"),
                    "strict": schema.get("strict"),
                },
            }
        elif response_format.get("type") == "json_object":
            responses["text"] = {"format": {"type": "json_object"}}

    return {"responses": responses}


def _map_finish_reason(reason: str | None) -> str:
    if reason in ("length", "max_tokens"):
        return "length"
    if reason in ("tool_calls", "tool_use"):
        return "tool_calls"
    if reason == "content_filter":
        return "content_filter"
    if reason == "error":
        return "error"
    return "stop"


def _extract_response_text(body: dict[str, Any]) -> str:
    output_text = body.get("output_text")
    if isinstance(output_text, str):
        return output_text
    text = ""
    for output in body.get("output") or []:
        for part in output.get("content") or []:
            if isinstance(part.get("text"), str):
                text += part["text"]
            elif isinstance(part.get("content"), str):
                text += part["content"]
    return text


def _extract_tool_calls(body: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = []
    for output in body.get("output") or []:
        if output.get("type") == "function_call":
            arguments = output.get("arguments")
            tool_calls.append({
                "id": output.get("call_id") or output.get("id") or generate_id("call"),
                "type": "function",
                "function": {
                    "name": output.get("name"),
                    "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments or {}),
                },
            })
    return tool_calls


def _translate_xai_response(body: dict[str, Any]) -> dict[str, Any]:
    if body.get("choices"):
        usage = body.get("usage") or {}
        return {
            "id": body.get("id") or generate_id(),
            "object": "chat.completion",
            "created": body.get("created") or int(time.time()),
            "model": f"xai/{body.get('model', 'unknown')}",
            "choices": body["choices"],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
        }

    tool_calls = _extract_tool_calls(body)
    message: dict[str, Any] = {"role": "assistant", "content": _extract_response_text(body)}
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = body.get("usage") or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))

    return {
        "id": body.get("id") or generate_id(),
        "object": "chat.completion",
        "created": body.get("created") or int(time.time()),
        "model": f"xai/{body.get('model', 'unknown')}",
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": "tool_calls" if tool_calls else _map_finish_reason(body.get("finish_reason") or body.get("status")),
        }],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": usage.get("total_tokens", input_tokens + output_tokens),
        },
    }


def _map_batch_status(data: dict[str, Any]) -> str:
    status = data.get("status") or data.get("processing_status") or (data.get("state") or {}).get("status")
    if status in ("cancelled", "canceled", "cancelling"):
        return "cancelled"
    if status in ("failed", "expired"):
        return "failed"
    if status in ("completed", "complete"):
        return "completed"
    if status in ("processing", "in_progress", "running"):
        return "processing"

    state = data.get("state") or {}
    pending = state.get("num_pending", 0)
    total = state.get("num_requests", 0)
    succeeded = state.get("num_success", 0)
    failed = state.get("num_error", 0) + state.get("num_cancelled", 0)

    if total > 0 and pending == 0:
        return "failed" if succeeded == 0 and failed > 0 else "completed"
    return "processing" if total > 0 else "pending"


class XAIBatchAdapter:
    """xAI native batch adapter."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=XAI_API_BASE,
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                timeout=get_default_timeout(),
            )
        return self._client

    async def _api_request(self, path: str, *, method: str = "GET", body: Any = None) -> httpx.Response:
        client = self._get_client()
        if method == "GET":
            res = await client.get(path)
        else:
            res = await client.request(method, path, json=body)

        if res.status_code >= 400:
            try:
                error_body = res.json()
            except Exception:
                error_body = {"message": res.reason_phrase}
            msg = error_body.get("error", {}).get("message") or error_body.get("message") or res.reason_phrase
            raise AnyModelError(
                502 if res.status_code >= 500 else res.status_code,
                msg or "xAI batch API error",
                {"provider_name": "xai", "raw": error_body},
            )
        return res

    async def create_batch(
        self, model: str, requests: list[dict[str, Any]], options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a batch to xAI's batch API."""
        name = options.get("name") if isinstance(options, dict) and isinstance(options.get("name"), str) else f"anymodel-batch-{int(time.time() * 1000)}"
        create_res = await self._api_request("/batches", method="POST", body={"name": name})
        batch = create_res.json()
        batch_id = batch.get("batch_id") or batch.get("id")
        if not batch_id:
            raise AnyModelError(502, "No batch id in xAI response", {"provider_name": "xai", "raw": batch})

        batch_requests = [
            {"batch_request_id": req.get("custom_id"), "batch_request": _translate_request(model, req)}
            for req in requests
        ]
        await self._api_request(f"/batches/{batch_id}/requests", method="POST", body={"batch_requests": batch_requests})

        return {"providerBatchId": batch_id, "metadata": {"model": model, "total_requests": len(requests)}}

    async def poll_batch(self, provider_batch_id: str) -> NativeBatchStatus:
        """Poll batch status."""
        res = await self._api_request(f"/batches/{provider_batch_id}")
        data = res.json()
        state = data.get("state") or {}
        total = state.get("num_requests", 0)
        failed = state.get("num_error", 0) + state.get("num_cancelled", 0)

        return NativeBatchStatus(
            status=_map_batch_status(data),
            total=total,
            completed=state.get("num_success", 0),
            failed=failed,
        )

    async def get_batch_results(self, provider_batch_id: str) -> list[dict[str, Any]]:
        """Download batch results, paginating through the results endpoint."""
        results: list[dict[str, Any]] = []
        pagination_token: str | None = None

        while True:
            params = {"limit": "100"}
            if pagination_token:
                params["pagination_token"] = pagination_token
            res = await self._api_request(f"/batches/{provider_batch_id}/results?{httpx.QueryParams(params)}")
            page = res.json()

            for item in page.get("results") or []:
                batch_result = item.get("batch_result") or {}
                response = batch_result.get("response")
                completion = (response or {}).get("chat_get_completion") or (response or {}).get("responses") or response
                error_message = item.get("error_message") or (batch_result.get("error") or {}).get("message")

                if completion and (
                    (response or {}).get("chat_get_completion")
                    or (response or {}).get("responses")
                    or completion.get("choices")
                    or completion.get("output")
                ):
                    results.append({
                        "custom_id": item.get("batch_request_id"),
                        "status": "success",
                        "response": _translate_xai_response(completion),
                        "error": None,
                    })
                else:
                    results.append({
                        "custom_id": item.get("batch_request_id"),
                        "status": "error",
                        "response": None,
                        "error": {
                            "code": (batch_result.get("error") or {}).get("code", 500),
                            "message": error_message or "Batch item error",
                        },
                    })

            pagination_token = page.get("pagination_token")
            if not pagination_token:
                break

        return results

    async def cancel_batch(self, provider_batch_id: str) -> None:
        """Cancel a batch."""
        await self._api_request(f"/batches/{provider_batch_id}:cancel", method="POST")


def create_xai_batch_adapter(api_key: str) -> XAIBatchAdapter:
    """Create an xAI native batch adapter."""
    return XAIBatchAdapter(api_key)
