"""Custom OpenAI-compatible provider adapter."""

from __future__ import annotations

from typing import Any

from anymodel.providers._openai import OpenAIAdapter


class CustomAdapter(OpenAIAdapter):
    """Wraps the OpenAI adapter with a custom provider name and base URL."""

    def __init__(self, provider_name: str, base_url: str, api_key: str = "", models: list[str] | None = None) -> None:
        super().__init__(api_key or "unused", base_url)
        self._provider_name = provider_name
        self._model_list = models

    @property
    def name(self) -> str:
        return self._provider_name

    def _re_prefix_id(self, id: str) -> str:
        if id and id.startswith("chatcmpl-"):
            return f"gen-{id[9:]}"
        return id if id.startswith("gen-") else f"gen-{id}"

    async def list_models(self) -> list[dict[str, Any]]:
        """Return configured models or fetch from the endpoint."""
        if self._model_list:
            return [
                {
                    "id": f"{self._provider_name}/{m}",
                    "name": m,
                    "created": 0,
                    "description": "",
                    "context_length": 128000,
                    "pricing": {"prompt": "0", "completion": "0"},
                    "architecture": {"modality": "text+image->text", "input_modalities": ["text", "image"],
                                     "output_modalities": ["text"], "tokenizer": "unknown"},
                    "top_provider": {"context_length": 128000, "max_completion_tokens": 16384, "is_moderated": False},
                    "supported_parameters": list(self._get_supported_params()),
                }
                for m in self._model_list
            ]
        # Try fetching from the endpoint
        try:
            return await super().list_models()
        except Exception:
            return []

    def _get_supported_params(self) -> frozenset[str]:
        from anymodel.providers._openai import SUPPORTED_PARAMS
        return SUPPORTED_PARAMS

    def supports_batch(self) -> bool:
        return False

    def translate_error(self, error: Exception) -> dict[str, Any]:
        result = super().translate_error(error)
        result["metadata"]["provider_name"] = self._provider_name
        return result


def create_custom_adapter(
    name: str,
    base_url: str,
    api_key: str = "",
    models: list[str] | None = None,
) -> CustomAdapter:
    """Create a custom OpenAI-compatible provider adapter."""
    return CustomAdapter(name, base_url, api_key, models)
