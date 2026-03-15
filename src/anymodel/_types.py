"""Types for anymodel — mirrors the OpenRouter/OpenAI API surface."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

# ─── Messages ────────────────────────────────────────────────────────────────

Role = Literal["system", "user", "assistant", "tool"]


class ImageURL(TypedDict, total=False):
    url: str
    detail: Literal["auto", "low", "high"]


class ContentPart(TypedDict, total=False):
    type: Literal["text", "image_url"]
    text: str
    image_url: ImageURL


class ToolCallFunction(TypedDict):
    name: str
    arguments: str


class ToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: ToolCallFunction


class Message(TypedDict, total=False):
    role: Role
    content: str | list[ContentPart]
    name: str
    tool_calls: list[ToolCall]
    tool_call_id: str


# ─── Tools ───────────────────────────────────────────────────────────────────


class FunctionDefinition(TypedDict, total=False):
    name: str
    description: str
    parameters: dict[str, Any]


class Tool(TypedDict):
    type: Literal["function"]
    function: FunctionDefinition


class ToolChoiceFunction(TypedDict):
    name: str


class ToolChoiceObject(TypedDict):
    type: Literal["function"]
    function: ToolChoiceFunction


ToolChoice = Literal["none", "auto", "required"] | ToolChoiceObject

# ─── Response Format ─────────────────────────────────────────────────────────


class TextFormat(TypedDict):
    type: Literal["text"]


class JsonObjectFormat(TypedDict):
    type: Literal["json_object"]


class JsonSchemaDefinition(TypedDict, total=False):
    name: str
    schema: dict[str, Any]
    strict: bool


class JsonSchemaFormat(TypedDict):
    type: Literal["json_schema"]
    json_schema: JsonSchemaDefinition


ResponseFormat = TextFormat | JsonObjectFormat | JsonSchemaFormat

# ─── Chat Completion Request ─────────────────────────────────────────────────


class ProviderPreferences(TypedDict, total=False):
    order: list[str]
    only: list[str]
    ignore: list[str]
    allow_fallbacks: bool
    require_parameters: bool
    sort: Literal["price", "throughput", "latency"]


class ChatCompletionRequest(TypedDict, total=False):
    # Required
    model: str
    messages: list[Message]
    # Standard optional
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    frequency_penalty: float
    presence_penalty: float
    repetition_penalty: float
    seed: int
    stop: str | list[str]
    stream: bool
    logprobs: bool
    top_logprobs: int
    response_format: ResponseFormat
    tools: list[Tool]
    tool_choice: ToolChoice
    user: str
    # Anymodel-specific
    models: list[str]
    route: Literal["fallback"]
    transforms: list[str]
    provider: ProviderPreferences


# ─── Chat Completion Response ────────────────────────────────────────────────

FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "error"]


class Usage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(TypedDict, total=False):
    index: int
    message: Message
    finish_reason: FinishReason
    logprobs: Any


class ChatCompletion(TypedDict):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


# ─── Streaming ───────────────────────────────────────────────────────────────


class ChunkDelta(TypedDict, total=False):
    role: Role
    content: str
    tool_calls: list[dict[str, Any]]


class ChunkChoice(TypedDict, total=False):
    index: int
    delta: ChunkDelta
    finish_reason: FinishReason | None
    logprobs: Any


class ChatCompletionChunk(TypedDict, total=False):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: list[ChunkChoice]
    usage: Usage


# ─── Models ──────────────────────────────────────────────────────────────────


class ModelPricing(TypedDict):
    prompt: str
    completion: str


class ModelArchitecture(TypedDict):
    modality: str
    input_modalities: list[str]
    output_modalities: list[str]
    tokenizer: str


class ModelTopProvider(TypedDict):
    context_length: int
    max_completion_tokens: int
    is_moderated: bool


class ModelInfo(TypedDict):
    id: str
    name: str
    created: int
    description: str
    context_length: int
    pricing: ModelPricing
    architecture: ModelArchitecture
    top_provider: ModelTopProvider
    supported_parameters: list[str]


# ─── Generation Stats ────────────────────────────────────────────────────────


class GenerationStats(TypedDict):
    id: str
    model: str
    provider_name: str
    total_cost: float
    tokens_prompt: int
    tokens_completion: int
    latency: float
    generation_time: float
    created_at: str
    finish_reason: FinishReason
    streamed: bool


# ─── Batch ───────────────────────────────────────────────────────────────────

BatchStatus = Literal["pending", "processing", "completed", "failed", "cancelled"]
BatchMode = Literal["native", "concurrent"]


class BatchRequestItem(TypedDict, total=False):
    custom_id: str
    messages: list[Message]
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    stop: str | list[str]
    response_format: ResponseFormat
    tools: list[Tool]
    tool_choice: ToolChoice


class BatchCreateOptions(TypedDict, total=False):
    temperature: float
    max_tokens: int
    top_p: float
    top_k: int
    stop: str | list[str]
    response_format: ResponseFormat
    tools: list[Tool]
    tool_choice: ToolChoice


class BatchCreateRequest(TypedDict, total=False):
    model: str
    requests: list[BatchRequestItem]
    options: BatchCreateOptions
    webhook: str


class BatchObject(TypedDict):
    id: str
    object: Literal["batch"]
    status: BatchStatus
    model: str
    provider_name: str
    batch_mode: BatchMode
    total: int
    completed: int
    failed: int
    created_at: str
    completed_at: str | None
    expires_at: str | None


class BatchError(TypedDict):
    code: int
    message: str


class BatchResultItem(TypedDict):
    custom_id: str
    status: Literal["success", "error"]
    response: ChatCompletion | None
    error: BatchError | None


class BatchUsageSummary(TypedDict):
    total_prompt_tokens: int
    total_completion_tokens: int
    estimated_cost: float


class BatchResults(TypedDict):
    id: str
    status: BatchStatus
    results: list[BatchResultItem]
    usage_summary: BatchUsageSummary


# ─── Config ──────────────────────────────────────────────────────────────────


class ProviderConfig(TypedDict, total=False):
    api_key: str
    default_model: str


class CustomProviderConfig(TypedDict, total=False):
    base_url: str
    api_key: str
    models: list[str]


class DefaultsConfig(TypedDict, total=False):
    temperature: float
    max_tokens: int
    retries: int
    timeout: float
    transforms: list[str]


class RoutingConfig(TypedDict, total=False):
    fallback_order: list[str]
    allow_fallbacks: bool


class BatchConfig(TypedDict, total=False):
    dir: str
    poll_interval: float
    concurrency_fallback: int
    retention_days: int


class IOConfig(TypedDict, total=False):
    read_concurrency: int
    write_concurrency: int


class AnyModelConfig(TypedDict, total=False):
    anthropic: ProviderConfig
    openai: ProviderConfig
    google: ProviderConfig
    mistral: ProviderConfig
    groq: ProviderConfig
    deepseek: ProviderConfig
    xai: ProviderConfig
    together: ProviderConfig
    fireworks: ProviderConfig
    perplexity: ProviderConfig
    ollama: ProviderConfig
    custom: dict[str, CustomProviderConfig]
    aliases: dict[str, str]
    defaults: DefaultsConfig
    routing: RoutingConfig
    batch: BatchConfig
    io: IOConfig


# ─── Errors ──────────────────────────────────────────────────────────────────


class AnyModelError(Exception):
    """Error from anymodel with HTTP-like status code and provider metadata."""

    def __init__(
        self,
        code: int,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": str(self),
                "metadata": self.metadata,
            }
        }
