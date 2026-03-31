# anymodel-py

OpenRouter-compatible LLM router with unified batch support. Self-hosted, zero fees.

Route requests across OpenAI, Anthropic, and Google with a single API. Add any OpenAI-compatible provider. Run as an SDK or standalone HTTP server.

## Install

```bash
pip install anymodel-py
```

## Quick Start

Set your API keys as environment variables:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GOOGLE_API_KEY=AIza...
```

### SDK Usage

```python
from anymodel import AnyModel

client = AnyModel()

response = await client.chat.completions.create(
    model="anthropic/claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Hello!"}],
)

print(response.choices[0].message.content)
```

### Streaming

```python
stream = await client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Write a haiku"}],
    stream=True,
)

async for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## Supported Providers

Set the env var and go. Models are auto-discovered from each provider's API.

| Provider | Env Var | Example Model |
|----------|---------|---------------|
| OpenAI | `OPENAI_API_KEY` | `openai/gpt-4o` |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic/claude-sonnet-4-6` |
| Google | `GOOGLE_API_KEY` | `google/gemini-2.5-pro` |
| Mistral | `MISTRAL_API_KEY` | `mistral/mistral-large-latest` |
| Groq | `GROQ_API_KEY` | `groq/llama-3.3-70b-versatile` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek/deepseek-chat` |
| xAI | `XAI_API_KEY` | `xai/grok-3` |
| Together | `TOGETHER_API_KEY` | `together/meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| Fireworks | `FIREWORKS_API_KEY` | `fireworks/accounts/fireworks/models/llama-v3p3-70b-instruct` |
| Perplexity | `PERPLEXITY_API_KEY` | `perplexity/sonar-pro` |
| Ollama | `OLLAMA_BASE_URL` | `ollama/llama3.3` |

Ollama runs locally with no API key — just set `OLLAMA_BASE_URL` (defaults to `http://localhost:11434/v1`).

## Model Naming

Models use `provider/model` format:

```
anthropic/claude-sonnet-4-6
openai/gpt-4o
google/gemini-2.5-pro
mistral/mistral-large-latest
groq/llama-3.3-70b-versatile
deepseek/deepseek-chat
xai/grok-3
perplexity/sonar-pro
ollama/llama3.3
```

### Flex Pricing (OpenAI)

Get 50% off OpenAI requests with flexible latency:

```python
response = await client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    service_tier="flex",
)
```

## Fallback Routing

Try multiple models in order. If one fails, the next is attempted:

```python
response = await client.chat.completions.create(
    model="",
    models=[
        "anthropic/claude-sonnet-4-6",
        "openai/gpt-4o",
        "google/gemini-2.5-pro",
    ],
    route="fallback",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Tool Calling

Works across all providers with a unified interface:

```python
response = await client.chat.completions.create(
    model="anthropic/claude-sonnet-4-6",
    messages=[{"role": "user", "content": "What's the weather in NYC?"}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            },
        },
    ],
    tool_choice="auto",
)

if response.choices[0].message.tool_calls:
    for call in response.choices[0].message.tool_calls:
        print(call.function.name, call.function.arguments)
```

## Structured Output

```python
response = await client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "List 3 colors"}],
    response_format={"type": "json_object"},
)
```

## Batch Processing

Process many requests with native provider batch APIs or concurrent fallback. OpenAI, Anthropic, and Google batches are processed server-side — OpenAI at 50% cost, Anthropic with async processing for up to 10K requests, Google at 50% cost via `batchGenerateContent`. Other providers fall back to concurrent execution automatically.

### Submit and wait

```python
results = await client.batches.create_and_poll(
    model="openai/gpt-4o-mini",
    requests=[
        {"custom_id": "req-1", "messages": [{"role": "user", "content": "Summarize AI"}]},
        {"custom_id": "req-2", "messages": [{"role": "user", "content": "Summarize ML"}]},
        {"custom_id": "req-3", "messages": [{"role": "user", "content": "Summarize NLP"}]},
    ],
)

for result in results.results:
    print(result.custom_id, result.response.choices[0].message.content)
```

### Submit now, check later

Submit a batch and get back an ID immediately — no need to keep the process running for native batches (OpenAI, Anthropic, Google):

```python
# Submit and get the batch ID
batch = await client.batches.create(
    model="anthropic/claude-haiku-4-5",
    requests=[
        {"custom_id": "req-1", "messages": [{"role": "user", "content": "Summarize AI"}]},
        {"custom_id": "req-2", "messages": [{"role": "user", "content": "Summarize ML"}]},
    ],
)
print(batch.id)          # "batch-abc123"
print(batch.batch_mode)  # "native" or "concurrent"

# Check status any time — even after a process restart
status = client.batches.get("batch-abc123")
print(status.status)  # "pending", "processing", "completed", "failed"

# Wait for results when you're ready (reconnects to provider API)
results = await client.batches.poll("batch-abc123")

# Or get results directly if already completed
results = client.batches.results("batch-abc123")
```

### List and cancel

```python
# List all batches on disk
all_batches = client.batches.list()
for b in all_batches:
    print(b.id, b.batch_mode, b.status, b.provider_name)

# Cancel a running batch (also cancels at the provider for native batches)
await client.batches.cancel("batch-abc123")
```

### BatchBuilder API

An ergonomic interface for building batches — just pass strings, and anymodel handles IDs, system prompt injection, and provider-specific formatting:

```python
batch = client.batches.open(
    model="anthropic/claude-sonnet-4-6",
    system="You are an expert.",
)

batch.add("What is an LLC?")
batch.add("How do I dissolve an LLC?")

await batch.submit()
results = await batch.poll()

print(results.succeeded)  # successful responses with per-item costs
print(results.failed)     # failed items
print(results.usage)      # aggregate usage and estimated_cost

# Retry failed items
retry_batch = batch.retry(results.failed)
await retry_batch.submit()
retry_results = await retry_batch.poll()
```

### Batch mode

Force concurrent execution instead of native batch APIs (useful when you want flex pricing on individual requests):

```python
results = await client.batches.create_and_poll(
    model="openai/gpt-4o",
    batch_mode="concurrent",  # skip native batch, run as individual requests
    requests=[
        {"custom_id": "req-1", "messages": [{"role": "user", "content": "Hello"}]},
    ],
)
```

### Service tier on batch requests

Use flex pricing on concurrent batches for 50% cost savings:

```python
results = await client.batches.create_and_poll(
    model="openai/gpt-4o",
    batch_mode="concurrent",
    service_tier="flex",  # flex pricing on each concurrent request
    requests=[
        {"custom_id": "req-1", "messages": [{"role": "user", "content": "Hello"}]},
    ],
)
```

### Poll logging

Enable console logging during batch polling to monitor progress:

```python
# Per-call option
results = await client.batches.create_and_poll(request, log_to_console=True)

# Or enable globally via environment variable
# ANYMODEL_BATCH_POLL_LOG=1
```

### Adaptive Concurrency

For concurrent batches, anymodel can automatically discover your provider's rate limit ceiling instead of using a fixed concurrency:

```python
client = AnyModel(
    batch={
        "concurrency_fallback": "auto",
    },
)
```

This uses TCP-style slow-start (exponential ramp: 5 → 10 → 20 → 40 → ...) to quickly find your ceiling, then switches to AIMD (additive increase / multiplicative decrease) for fine-tuning. It reads `x-ratelimit-remaining-requests` headers proactively and backs off on 429s — so an OpenAI Tier 4 account at 10,000 RPM will ramp to ~160 concurrent in about 155 requests instead of being stuck at 5.

Use `concurrency_max` to set a hard ceiling — useful when multiple batch jobs share the same API key:

```python
client = AnyModel(
    batch={
        "concurrency_fallback": "auto",
        "concurrency_max": 50,  # each job caps at 50, two jobs = 100 total
    },
)
```

### Batch configuration

```python
client = AnyModel(
    batch={
        "poll_interval": 10000,          # default poll interval in ms (default: 5000)
        "concurrency_fallback": 10,      # concurrent request limit for non-native providers (default: 5)
        # "concurrency_fallback": "auto", # or auto-discover from provider rate limits
        # "concurrency_max": 50,          # hard ceiling for auto mode
    },
    io={
        "read_concurrency": 30,   # concurrent file reads (default: 20)
        "write_concurrency": 15,  # concurrent file writes (default: 10)
    },
)

# Override poll interval per call
results = await client.batches.create_and_poll(
    request,
    interval=3000,  # poll every 3s for this batch
    on_progress=lambda batch: print(f"{batch.completed}/{batch.total} done"),
)
```

Batches are persisted to `./.anymodel/batches/` in the current working directory and survive process restarts.

### Automatic max_tokens

When `max_tokens` isn't set on a batch request, anymodel automatically calculates a safe value per-request based on the estimated input size and the model's context window. This prevents truncated responses and context overflow errors without requiring you to hand-tune each request in a large batch. The estimation uses a ~4 chars/token heuristic with a 5% safety margin — conservative enough to avoid overflows, lightweight enough to skip tokenizer dependencies.

## Models Endpoint

```python
models = await client.models.list()
anthropic_models = await client.models.list(provider="anthropic")
```

## Generation Stats

```python
response = await client.chat.completions.create(...)
stats = client.generation.get(response.id)
print(stats.latency, stats.tokens_prompt, stats.tokens_completion)
print(stats.total_cost)  # auto-calculated from bundled pricing data
```

### Auto Pricing / Cost Calculation

Pricing for 323 models is baked in at build time from OpenRouter — always current as of last publish. Costs are calculated automatically from token usage with no configuration needed.

```python
# Per-request cost on GenerationStats
stats = client.generation.get(response.id)
print(stats.total_cost)  # e.g. 0.0023

# Batch-level cost on BatchUsageSummary
results = await client.batches.create_and_poll(request)
print(results.usage.estimated_cost)  # total across all requests

# Native batch pricing is automatically 50% off
# Utility functions also exported
from anymodel import get_model_pricing, calculate_cost, PRICING_AS_OF, PRICING_MODEL_COUNT
```

## Configuration

### Programmatic

```python
client = AnyModel(
    anthropic={"api_key": "sk-ant-..."},
    openai={"api_key": "sk-..."},
    google={"api_key": "AIza..."},
    aliases={
        "default": "anthropic/claude-sonnet-4-6",
        "fast": "anthropic/claude-haiku-4-5",
        "smart": "anthropic/claude-opus-4-6",
    },
    defaults={
        "temperature": 0.7,
        "max_tokens": 4096,
        "retries": 2,
        "timeout": 120,  # HTTP timeout in seconds (default: 120 = 2 min, flex: 600 = 10 min)
    },
)

# Use aliases as model names
response = await client.chat.completions.create(
    model="fast",
    messages=[{"role": "user", "content": "Quick answer"}],
)
```

### Config File

Create `anymodel.config.json` in your project root:

```json
{
  "anthropic": {
    "apiKey": "${ANTHROPIC_API_KEY}"
  },
  "aliases": {
    "default": "anthropic/claude-sonnet-4-6",
    "fast": "anthropic/claude-haiku-4-5"
  },
  "defaults": {
    "temperature": 0.7,
    "max_tokens": 4096
  },
  "batch": {
    "pollInterval": 5000,
    "concurrencyFallback": 5
  },
  "io": {
    "readConcurrency": 20,
    "writeConcurrency": 10
  }
}
```

`${ENV_VAR}` references are interpolated from environment variables.

### Config Resolution Order

1. Programmatic options (highest priority)
2. Local `anymodel.config.json`
3. Global `~/.anymodel/config.json`
4. Environment variables (lowest priority)

Configs are deep-merged, not replaced.

## Custom Providers

Add any OpenAI-compatible endpoint:

```python
client = AnyModel(
    custom={
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "models": ["llama3.3", "mistral"],
        },
        "together": {
            "base_url": "https://api.together.xyz/v1",
            "api_key": "your-key",
        },
    },
)

response = await client.chat.completions.create(
    model="ollama/llama3.3",
    messages=[{"role": "user", "content": "Hello from Ollama"}],
)
```

## Provider Preferences

Control which providers are used and in what order:

```python
response = await client.chat.completions.create(
    model="",
    models=["anthropic/claude-sonnet-4-6", "openai/gpt-4o", "google/gemini-2.5-pro"],
    route="fallback",
    provider={
        "order": ["anthropic", "openai"],
        "ignore": ["google"],
    },
    messages=[{"role": "user", "content": "Hello"}],
)
```

## Transforms

Automatically truncate long conversations to fit within context windows:

```python
response = await client.chat.completions.create(
    model="anthropic/claude-sonnet-4-6",
    messages=very_long_conversation,
    transforms=["middle-out"],
)
```

`middle-out` preserves the system prompt and most recent messages, removing from the middle.

## Server Mode

Run as a standalone HTTP server compatible with the OpenAI SDK:

```bash
anymodel serve --port 4141
```

Then point any OpenAI-compatible client at it:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4141/api/v1",
    api_key="unused",
)

response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Hello via server"}],
)
```

### Server Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/chat/completions` | Chat completion (streaming supported) |
| GET | `/api/v1/models` | List available models |
| GET | `/api/v1/generation/:id` | Get generation stats |
| POST | `/api/v1/batches` | Create a batch |
| GET | `/api/v1/batches` | List batches |
| GET | `/api/v1/batches/:id` | Get batch status |
| GET | `/api/v1/batches/:id/results` | Get batch results |
| POST | `/api/v1/batches/:id/cancel` | Cancel a batch |
| GET | `/health` | Health check |

## Examples

See [`examples/basic.py`](examples/basic.py) for runnable demos of completions, streaming, tool calling, fallback routing, batch processing, and generation stats.

```bash
# Run all examples
python examples/basic.py

# Run a specific example
python examples/basic.py stream
python examples/basic.py tools
python examples/basic.py batch
```

## Built-in Resilience

- **Retries**: Automatic retry with exponential backoff on 429/502/503 errors (configurable via `defaults.retries`)
- **Rate limit tracking**: Per-provider rate limit state from response headers, automatically skips rate-limited providers during fallback routing
- **Adaptive concurrency**: Auto mode discovers your provider's actual rate limit ceiling using TCP-style slow-start + AIMD, reading `x-ratelimit-remaining-requests` headers proactively
- **Parameter translation**: `max_tokens` automatically sent as `max_completion_tokens` for newer OpenAI models (gpt-4o, o1, o3, gpt-5-mini). Unsupported parameters stripped before forwarding.
- **Smart batch defaults**: Automatic `max_tokens` estimation per-request in batches — calculates safe values from input size and model context limits, preventing truncation and overflow without manual tuning
- **Memory-efficient batching**: Concurrent batch requests are streamed from disk — only N requests (default 5) are in-flight at a time, making 10K+ request batches safe without memory spikes
- **High-volume IO**: All batch file operations use concurrency-limited async queues with atomic durable writes (temp file + fsync + rename) to prevent corruption on crash. Defaults: 20 concurrent reads, 10 concurrent writes — configurable via `io.read_concurrency` and `io.write_concurrency`

## Roadmap

- [ ] **A/B testing** — split routing (% traffic to each model) and compare mode (same request to multiple models, return all responses with stats)
- [x] **Cost tracking** — per-request and aggregate cost calculation from bundled pricing data (323 models from OpenRouter)
- [ ] **Caching** — response caching with configurable TTL for identical requests
- [x] **Native batch APIs** — OpenAI Batch API (JSONL upload, 50% cost), Anthropic Message Batches (10K requests, async), and Google Gemini Batch (50% cost). Auto-detects provider and routes to native API, falls back to concurrent for other providers
- [x] **Adaptive concurrency** — auto-discover provider rate limit ceilings via TCP slow-start + AIMD, with hard cap support for multi-job workloads
- [ ] **Result export** — `save_results()` to write batch results to a configurable output directory
- [ ] **Prompt logging** — optional request/response logging for debugging and evaluation

## See Also

| Package | Description |
|---|---|
| [anymodel](https://github.com/probeo-io/anymodel) | TypeScript version of this package |
| [anymodel-go](https://github.com/probeo-io/anymodel-go) | Go version of this package |
| [@probeo/anyserp](https://github.com/probeo-io/anyserp) | Unified SERP API router for TypeScript |
| [@probeo/workflow](https://github.com/probeo-io/workflow) | Stage-based pipeline engine for TypeScript |

## License

MIT
