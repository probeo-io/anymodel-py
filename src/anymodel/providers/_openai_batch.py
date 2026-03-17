"""OpenAI batch adapter."""

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

OPENAI_API_BASE = "https://api.openai.com/v1"


def _re_prefix_id(id_: str) -> str:
    """Re-prefix OpenAI completion IDs to use gen- prefix."""
    if id_ and id_.startswith("chatcmpl-"):
        return f"gen-{id_[9:]}"
    return id_ if id_.startswith("gen-") else f"gen-{id_}"


def _translate_openai_response(body: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI ChatCompletion to our normalized format."""
    return {
        "id": _re_prefix_id(body.get("id") or generate_id()),
        "object": "chat.completion",
        "created": body.get("created", int(time.time())),
        "model": f"openai/{body.get('model', 'unknown')}",
        "choices": body.get("choices", []),
        "usage": body.get("usage", {}),
    }


def _map_status(openai_status: str) -> str:
    """Map OpenAI batch status to our normalized status."""
    return {
        "validating": "processing",
        "finalizing": "processing",
        "in_progress": "processing",
        "completed": "completed",
        "failed": "failed",
        "expired": "failed",
        "cancelled": "cancelled",
        "cancelling": "cancelled",
    }.get(openai_status, "pending")


def _build_jsonl(model: str, requests: list[dict[str, Any]]) -> str:
    """Build JSONL content for an OpenAI batch upload."""
    lines = []
    for req in requests:
        body: dict[str, Any] = {
            "model": model,
            "messages": req.get("messages", []),
        }
        # service_tier intentionally omitted — native batch already gets 50% off
        for key in ("max_tokens", "temperature", "top_p", "stop",
                     "response_format", "tools", "tool_choice"):
            if key in req:
                body[key] = req[key]

        if "max_tokens" not in body:
            body["max_tokens"] = resolve_max_tokens(model, req.get("messages", []))

        lines.append(json.dumps({
            "custom_id": req.get("custom_id", f"request-{len(lines)}"),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }))
    return "\n".join(lines)


class OpenAIBatchAdapter:
    """OpenAI native batch adapter."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=OPENAI_API_BASE,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
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
        files: dict[str, Any] | None = None,
        data: dict[str, str] | None = None,
    ) -> httpx.Response:
        client = self._get_client()

        if files:
            res = await client.post(path, files=files, data=data)
        elif method == "GET":
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
                msg or "OpenAI batch API error",
                {"provider_name": "openai", "raw": error_body},
            )

        return res

    async def create_batch(
        self,
        model: str,
        requests: list[dict[str, Any]],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a batch to OpenAI's batch API."""
        # 1. Build JSONL content
        jsonl_content = _build_jsonl(model, requests)

        # 2. Upload file
        upload_res = await self._api_request(
            "/files",
            method="POST",
            files={"file": ("batch_input.jsonl", jsonl_content.encode(), "application/jsonl")},
            data={"purpose": "batch"},
        )
        file_data = upload_res.json()
        input_file_id = file_data["id"]

        # 3. Create batch
        batch_body: dict[str, Any] = {
            "input_file_id": input_file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        }
        if options and options.get("metadata"):
            batch_body["metadata"] = options["metadata"]

        batch_res = await self._api_request(
            "/batches",
            method="POST",
            body=batch_body,
        )
        batch_data = batch_res.json()

        return {
            "providerBatchId": batch_data["id"],
            "metadata": {
                "input_file_id": input_file_id,
                "openai_status": batch_data.get("status"),
            },
        }

    async def poll_batch(self, provider_batch_id: str) -> NativeBatchStatus:
        """Poll batch status."""
        res = await self._api_request(f"/batches/{provider_batch_id}")
        data = res.json()

        request_counts = data.get("request_counts", {})

        return NativeBatchStatus(
            status=_map_status(data.get("status", "pending")),
            total=request_counts.get("total", 0),
            completed=request_counts.get("completed", 0),
            failed=request_counts.get("failed", 0),
        )

    async def get_batch_results(self, provider_batch_id: str) -> list[dict[str, Any]]:
        """Download batch results."""
        # Get batch to find output file
        batch_res = await self._api_request(f"/batches/{provider_batch_id}")
        batch_data = batch_res.json()

        results: list[dict[str, Any]] = []

        # Download output file
        output_file_id = batch_data.get("output_file_id")
        if output_file_id:
            output_res = await self._api_request(f"/files/{output_file_id}/content")
            output_text = output_res.text

            for line in output_text.strip().split("\n"):
                if not line:
                    continue
                item = json.loads(line)

                if item.get("response", {}).get("status_code") == 200:
                    results.append({
                        "custom_id": item["custom_id"],
                        "status": "success",
                        "response": _translate_openai_response(item["response"]["body"]),
                        "error": None,
                    })
                else:
                    results.append({
                        "custom_id": item["custom_id"],
                        "status": "error",
                        "response": None,
                        "error": {
                            "code": item.get("response", {}).get("status_code", 500),
                            "message": (
                                item.get("error", {}).get("message")
                                or item.get("response", {}).get("body", {}).get("error", {}).get("message")
                                or "Unknown error"
                            ),
                        },
                    })

        # Download error file
        error_file_id = batch_data.get("error_file_id")
        if error_file_id:
            error_res = await self._api_request(f"/files/{error_file_id}/content")
            error_text = error_res.text

            existing_ids = {r["custom_id"] for r in results}
            for line in error_text.strip().split("\n"):
                if not line:
                    continue
                item = json.loads(line)
                if item.get("custom_id") not in existing_ids:
                    results.append({
                        "custom_id": item["custom_id"],
                        "status": "error",
                        "response": None,
                        "error": {
                            "code": item.get("response", {}).get("status_code", 500),
                            "message": item.get("error", {}).get("message", "Batch item error"),
                        },
                    })

        return results

    async def cancel_batch(self, provider_batch_id: str) -> None:
        """Cancel a batch."""
        await self._api_request(f"/batches/{provider_batch_id}/cancel", method="POST")


def create_openai_batch_adapter(api_key: str) -> OpenAIBatchAdapter:
    """Create an OpenAI batch adapter."""
    return OpenAIBatchAdapter(api_key)
