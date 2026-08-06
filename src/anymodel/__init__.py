"""anymodel — OpenRouter-compatible LLM router with unified batch support."""

from anymodel._cache import create_prompt_cache_key, with_prompt_cache
from anymodel._client import AnyModel
from anymodel._types import (
    AnyModelError,
    BatchCreateRequest,
    BatchMode,
    BatchObject,
    BatchRequestItem,
    BatchResultItem,
    BatchResults,
    BatchStatus,
    BatchUsageSummary,
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionWithMeta,
    FinishReason,
    GenerationStats,
    Message,
    ModelInfo,
    PromptCacheOptions,
    ReasoningOptions,
    ResponseMeta,
    Role,
    Tool,
    ToolCall,
    ToolChoice,
    Usage,
)
from anymodel.batch._builder import BatchBuilder
from anymodel.utils._adaptive_concurrency import AdaptiveConcurrencyController
from anymodel.utils._fs_io import configure_fs_io

try:
    from anymodel.generated.pricing import (
        PRICING_AS_OF,
        PRICING_MODEL_COUNT,
        calculate_cost,
        calculate_provider_cost,
        get_model_pricing,
    )
except ImportError:
    PRICING_AS_OF: str = ""  # type: ignore[no-redef]
    PRICING_MODEL_COUNT: int = 0  # type: ignore[no-redef]

    def get_model_pricing(model_id: str) -> dict[str, float] | None:
        return None

    def calculate_cost(model_id: str, prompt_tokens: int, completion_tokens: int, cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float:
        return 0.0

    def calculate_provider_cost(model: str, prompt_tokens: int, completion_tokens: int, **kwargs: object) -> dict[str, object]:  # type: ignore[misc]
        return {"estimated_cost": 0.0, "multiplier": 1.0, "pricing": None}

__version__ = "0.7.0"

__all__ = [
    "AdaptiveConcurrencyController",
    "AnyModel",
    "AnyModelError",
    "BatchBuilder",
    "configure_fs_io",
    # Prompt caching
    "create_prompt_cache_key",
    "with_prompt_cache",
    # Pricing
    "calculate_cost",
    "calculate_provider_cost",
    "get_model_pricing",
    "PRICING_AS_OF",
    "PRICING_MODEL_COUNT",
    # Types
    "BatchCreateRequest",
    "BatchMode",
    "BatchObject",
    "BatchRequestItem",
    "BatchResultItem",
    "BatchResults",
    "BatchStatus",
    "BatchUsageSummary",
    "ChatCompletion",
    "ChatCompletionChunk",
    "ChatCompletionRequest",
    "ChatCompletionWithMeta",
    "FinishReason",
    "GenerationStats",
    "Message",
    "ModelInfo",
    "PromptCacheOptions",
    "ReasoningOptions",
    "ResponseMeta",
    "Role",
    "Tool",
    "ToolCall",
    "ToolChoice",
    "Usage",
]
