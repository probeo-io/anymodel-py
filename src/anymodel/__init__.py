"""anymodel — OpenRouter-compatible LLM router with unified batch support."""

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

__version__ = "0.3.0"

__all__ = [
    "AnyModel",
    "AnyModelError",
    "configure_fs_io",
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
