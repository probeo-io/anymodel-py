"""anymodel — OpenRouter-compatible LLM router with unified batch support."""

from anymodel._client import AnyModel
from anymodel.batch._builder import BatchBuilder
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
    FinishReason,
    GenerationStats,
    Message,
    ModelInfo,
    Role,
    Tool,
    ToolCall,
    ToolChoice,
    Usage,
)
from anymodel.utils._fs_io import configure_fs_io

try:
    from anymodel.generated.pricing import (
        PRICING_AS_OF,
        calculate_cost,
        get_model_pricing,
    )
except ImportError:
    PRICING_AS_OF: str = ""  # type: ignore[no-redef]

    def get_model_pricing(model_id: str):  # type: ignore[misc]
        return None

    def calculate_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> float:  # type: ignore[misc]
        return 0.0

__version__ = "0.5.0"

__all__ = [
    "AnyModel",
    "AnyModelError",
    "BatchBuilder",
    "configure_fs_io",
    # Pricing
    "calculate_cost",
    "get_model_pricing",
    "PRICING_AS_OF",
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
    "FinishReason",
    "GenerationStats",
    "Message",
    "ModelInfo",
    "Role",
    "Tool",
    "ToolCall",
    "ToolChoice",
    "Usage",
]
