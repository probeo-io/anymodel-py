# anymodel-py

[![PyPI version](https://img.shields.io/pypi/v/anymodel-py)](https://pypi.org/project/anymodel-py/)
[![PyPI downloads](https://img.shields.io/pypi/dm/anymodel-py)](https://pypi.org/project/anymodel-py/)
[![License](https://img.shields.io/pypi/l/anymodel-py)](https://github.com/probeo-io/anymodel-py/blob/main/LICENSE)
[![Python versions](https://img.shields.io/pypi/pyversions/anymodel-py)](https://pypi.org/project/anymodel-py/)
[![CI](https://github.com/probeo-io/anymodel-py/actions/workflows/ci.yml/badge.svg)](https://github.com/probeo-io/anymodel-py/actions/workflows/ci.yml)

OpenRouter-compatible LLM router with unified batch support. Self-hosted, zero fees.

Route requests across OpenAI, Anthropic, and Google with a single API. Add any OpenAI-compatible provider. Run as an SDK or standalone HTTP server.

## Why anymodel?

One SDK, 11+ providers, no vendor lock-in. Swap models by changing a string, not rewriting your integration.

Self-hosted and transparent. Your API keys, your infrastructure, zero routing fees. Unlike OpenRouter, nothing passes through a third party.

Native batch APIs (OpenAI, Anthropic, Google) run at 50% cost with zero config. For other providers, anymodel falls back to concurrent execution automatically.

Zero provider SDK dependencies. The only runtime requirement is httpx. Server mode is drop-in compatible with the OpenAI SDK, so existing code works without changes.

## Install

```bash
pip install anymodel-py
```

For server mode, install with the `server` extra:

```bash
pip install anymodel-py[server]
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
| xAI | `XAI_API_KEY` | `xai/grok-4` |
| Together | `TOGETHER_API_KEY` | `together/meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| Fireworks | `FIREWORKS_API_KEY` | `fireworks/accounts/fireworks/models/llama-v3p3-70b-instruct` |
| Perplexity | `PERPLEXITY_API_KEY` | `perplexity/sonar-pro` |
| Ollama | `OLLAMA_BASE_URL` | `ollama/llama3.3` |

Ollama runs locally with no API key. Just set `OLLAMA_BASE_URL` (defaults to `http://localhost:11434/v1`).

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

## Batch Processing

Process many requests with native provider batch APIs or concurrent fallback. OpenAI, Anthropic, and Google batches run server-side at up to 50% cost. Other providers fall back to concurrent execution automatically.

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

Native batches (OpenAI, Anthropic, Google) are processed asynchronously on the provider's infrastructure. For all other providers, anymodel runs requests concurrently with adaptive rate limiting. See [Advanced Usage](docs/ADVANCED.md) for BatchBuilder, submit-now-check-later, adaptive concurrency, and batch configuration.

## Server Mode

Run as a standalone HTTP server compatible with the OpenAI SDK:

```bash
anymodel serve --port 4141
```

Point any OpenAI-compatible client at `http://localhost:4141/api/v1`. See [Advanced Usage](docs/ADVANCED.md) for the full endpoint reference.

## Configuration

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
        "timeout": 120,
    },
)

# Use aliases as model names
response = await client.chat.completions.create(
    model="fast",
    messages=[{"role": "user", "content": "Quick answer"}],
)
```

Configuration can also be loaded from an `anymodel.config.json` file. Both camelCase and snake_case keys are accepted (`apiKey` or `api_key`, `pollInterval` or `poll_interval`). See [Advanced Usage](docs/ADVANCED.md) for config file details and resolution order.

## Built-in Resilience

- **Retries**: Automatic retry with exponential backoff on 429/502/503 errors (configurable via `defaults.retries`)
- **Rate limit tracking**: Per-provider rate limit state from response headers. Automatically skips rate-limited providers during fallback routing.
- **Adaptive concurrency**: Auto mode discovers your provider's actual rate limit ceiling using TCP-style slow-start + AIMD, reading `x-ratelimit-remaining-requests` headers proactively.
- **Parameter translation**: `max_tokens` automatically sent as `max_completion_tokens` for newer OpenAI models. Unsupported parameters stripped before forwarding.
- **Smart batch defaults**: Automatic `max_tokens` estimation per-request in batches. Calculates safe values from input size and model context limits, preventing truncation and overflow without manual tuning.
- **Memory-efficient batching**: Concurrent batch requests are streamed from disk. Only N requests (default 5) are in-flight at a time, making 10K+ request batches safe without memory spikes.
- **High-volume IO**: All batch file operations use concurrency-limited async queues with atomic durable writes (temp file + fsync + rename) to prevent corruption on crash.

See [Advanced Usage](docs/ADVANCED.md) for tool calling, structured output, BatchBuilder, adaptive concurrency, custom providers, transforms, generation stats, auto pricing, and more.

## See Also

| Package | Description |
|---|---|
| [anymodel](https://github.com/probeo-io/anymodel) | TypeScript version of this package |
| [anymodel-go](https://github.com/probeo-io/anymodel-go) | Go version of this package |
| [@probeo/anyserp](https://github.com/probeo-io/anyserp) | Unified SERP API router for TypeScript |
| [@probeo/workflow](https://github.com/probeo-io/workflow) | Stage-based pipeline engine for TypeScript |

## Support

If anymodel is useful to you, consider giving it a star. It helps others discover the project.

## License

MIT
