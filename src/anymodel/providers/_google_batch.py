"""Google Gemini batch adapter."""

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

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def _map_finish_reason(reason: str) -> str:
    return {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
    }.get(reason, "stop")


def _translate_request_to_gemini(req: dict[str, Any], model: str = "") -> dict[str, Any]:
    """Convert an OpenAI-style batch request item to Gemini format."""
    body: dict[str, Any] = {}

    messages = req.get("messages", [])
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system_msgs = [m for m in messages if m.get("role") != "system"]

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

        if isinstance(content, str):
            contents.append({"role": gemini_role, "parts": [{"text": content}]})
        elif isinstance(content, list):
            parts = []
            for part in content:
                if part.get("type") == "text":
                    parts.append({"text": part.get("text", "")})
            contents.append({"role": gemini_role, "parts": parts or [{"text": ""}]})

    body["contents"] = contents

    # Generation config
    gen_config: dict[str, Any] = {}
    if "temperature" in req:
        gen_config["temperature"] = req["temperature"]
    if "max_tokens" in req:
        gen_config["maxOutputTokens"] = req["max_tokens"]
    if "top_p" in req:
        gen_config["topP"] = req["top_p"]
    if "top_k" in req:
        gen_config["topK"] = req["top_k"]
    if "stop" in req:
        stop = req["stop"]
        gen_config["stopSequences"] = stop if isinstance(stop, list) else [stop]

    response_format = req.get("response_format")
    if response_format:
        if response_format.get("type") == "json_object":
            gen_config["responseMimeType"] = "application/json"
        elif response_format.get("type") == "json_schema":
            gen_config["responseMimeType"] = "application/json"
            schema = response_format.get("json_schema", {}).get("schema")
            if schema:
                gen_config["responseSchema"] = schema

    if "max_tokens" not in req:
        gen_config["maxOutputTokens"] = resolve_max_tokens(
            model, req.get("messages", []),
        )

    if gen_config:
        body["generationConfig"] = gen_config

    # Tools
    tools = req.get("tools")
    if tools:
        body["tools"] = [{"functionDeclarations": [
            {
                "name": t["function"]["name"],
                "description": t["function"].get("description", ""),
                "parameters": t["function"].get("parameters", {}),
            }
            for t in tools
        ]}]

        tool_choice = req.get("tool_choice")
        if tool_choice:
            if tool_choice == "auto":
                body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
            elif tool_choice == "required":
                body["toolConfig"] = {"functionCallingConfig": {"mode": "ANY"}}
            elif tool_choice == "none":
                body["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}
            elif isinstance(tool_choice, dict):
                body["toolConfig"] = {
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [tool_choice["function"]["name"]],
                    },
                }

    return body


def _translate_gemini_response(response: dict[str, Any], model: str) -> dict[str, Any]:
    """Convert a Gemini response to OpenAI ChatCompletion format."""
    candidates = response.get("candidates", [{}])
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

    finish_reason = (
        "tool_calls" if tool_calls
        else _map_finish_reason(candidate.get("finishReason", "STOP"))
    )

    usage_meta = response.get("usageMetadata", {})
    prompt_tokens = usage_meta.get("promptTokenCount", 0)
    completion_tokens = usage_meta.get("candidatesTokenCount", 0)

    return {
        "id": generate_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": f"google/{model}",
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _map_batch_state(state: str) -> str:
    return {
        "JOB_STATE_PENDING": "pending",
        "JOB_STATE_RUNNING": "processing",
        "JOB_STATE_SUCCEEDED": "completed",
        "JOB_STATE_FAILED": "failed",
        "JOB_STATE_CANCELLED": "cancelled",
        "JOB_STATE_EXPIRED": "failed",
    }.get(state, "pending")


class GoogleBatchAdapter:
    """Google Gemini native batch adapter."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=GEMINI_API_BASE,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self._api_key,
                },
                timeout=get_default_timeout(),
            )
        return self._client

    async def _api_request(
        self, path: str, *, method: str = "GET", body: Any = None,
    ) -> dict[str, Any]:
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
            msg = error_body.get("error", {}).get("message") or res.reason_phrase
            raise AnyModelError(
                502 if res.status_code >= 500 else res.status_code,
                msg or "Google batch API error",
                {"provider_name": "google", "raw": error_body},
            )

        return res.json()

    async def create_batch(
        self,
        model: str,
        requests: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a batch to Gemini's batchGenerateContent endpoint."""
        batch_requests = [
            {
                "request": _translate_request_to_gemini(req, model),
                "metadata": {"key": req.get("custom_id", f"request-{i}")},
            }
            for i, req in enumerate(requests)
        ]

        data = await self._api_request(
            f"/models/{model}:batchGenerateContent",
            method="POST",
            body={
                "batch": {
                    "display_name": f"anymodel-batch-{int(time.time())}",
                    "input_config": {
                        "requests": {
                            "requests": batch_requests,
                        },
                    },
                },
            },
        )

        batch_name = data.get("name") or (data.get("batch") or {}).get("name")
        if not batch_name:
            raise AnyModelError(
                502, "No batch name in Google response",
                {"provider_name": "google", "raw": data},
            )

        return {
            "providerBatchId": batch_name,
            "metadata": {"model": model, "total_requests": len(requests)},
        }

    async def poll_batch(self, provider_batch_id: str) -> NativeBatchStatus:
        """Poll batch status."""
        data = await self._api_request(f"/{provider_batch_id}")

        state = data.get("state", "JOB_STATE_PENDING")
        status = _map_batch_state(state)

        total_count = data.get("totalCount") or (data.get("metadata") or {}).get("total_requests", 0)
        success_count = data.get("succeededCount", 0)
        failed_count = data.get("failedCount", 0)

        return NativeBatchStatus(
            status=status,
            total=total_count or (success_count + failed_count),
            completed=success_count,
            failed=failed_count,
        )

    async def get_batch_results(self, provider_batch_id: str) -> list[dict[str, Any]]:
        """Download batch results."""
        data = await self._api_request(f"/{provider_batch_id}")

        results: list[dict[str, Any]] = []
        model = (data.get("metadata") or {}).get("model", "unknown")

        # Check for inline responses
        inlined = (data.get("response") or {}).get("inlinedResponses")
        if inlined:
            for item in inlined:
                custom_id = (item.get("metadata") or {}).get("key", f"request-{len(results)}")
                if item.get("response"):
                    results.append({
                        "custom_id": custom_id,
                        "status": "success",
                        "response": _translate_gemini_response(item["response"], model),
                        "error": None,
                    })
                elif item.get("error"):
                    results.append({
                        "custom_id": custom_id,
                        "status": "error",
                        "response": None,
                        "error": {
                            "code": item["error"].get("code", 500),
                            "message": item["error"].get("message", "Batch item failed"),
                        },
                    })
            return results

        # Check for file-based results
        responses_file = (
            (data.get("response") or {}).get("responsesFileName")
            or (data.get("outputConfig") or {}).get("file_name")
        )

        if responses_file:
            download_url = f"{GEMINI_API_BASE}/{responses_file}:download?alt=media"
            client = self._get_client()
            file_res = await client.get(
                download_url,
                headers={"x-goog-api-key": self._api_key},
            )

            if file_res.status_code >= 400:
                raise AnyModelError(
                    502, "Failed to download batch results file",
                    {"provider_name": "google"},
                )

            for line in file_res.text.strip().split("\n"):
                if not line:
                    continue
                item = json.loads(line)
                custom_id = (
                    item.get("key")
                    or (item.get("metadata") or {}).get("key")
                    or f"request-{len(results)}"
                )
                if item.get("response"):
                    results.append({
                        "custom_id": custom_id,
                        "status": "success",
                        "response": _translate_gemini_response(item["response"], model),
                        "error": None,
                    })
                elif item.get("error"):
                    results.append({
                        "custom_id": custom_id,
                        "status": "error",
                        "response": None,
                        "error": {
                            "code": item["error"].get("code", 500),
                            "message": item["error"].get("message", "Batch item failed"),
                        },
                    })

        return results

    async def cancel_batch(self, provider_batch_id: str) -> None:
        """Cancel a batch."""
        await self._api_request(f"/{provider_batch_id}:cancel", method="POST")


def create_google_batch_adapter(api_key: str) -> GoogleBatchAdapter:
    """Create a Google Gemini batch adapter."""
    return GoogleBatchAdapter(api_key)
