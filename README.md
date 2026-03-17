# anymodel

OpenRouter-compatible LLM router with unified batch support for Python. Self-hosted, zero fees.

Route requests across OpenAI, Anthropic, and Google with a single API. Add any OpenAI-compatible provider. Run as an SDK or standalone HTTP server.

## Install

```bash
pip install anymodel
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
import asyncio
from anymodel import AnyModel

async def main():
    client = AnyModel()

    response = await client.chat.completions.create(
        model="anthropic/claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    print(response["choices"][0]["message"]["content"])

asyncio.run(main())
```

### Streaming

```python
stream = await client.chat.completions.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "Write a haiku"}],
    stream=True,
)

async for chunk in stream:
    content = chunk["choices"][0].get("delta", {}).get("content", "")
    print(content, end="", flush=True)
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

```python
response = await client.chat.completions.create(
    model="anthropic/claude-sonnet-4-6",
    messages=[{"role": "user", "content": "What's the weather in NYC?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }],
    tool_choice="auto",
)

for call in response["choices"][0]["message"].get("tool_calls", []):
    print(call["function"]["name"], call["function"]["arguments"])
```

## Batch Processing

Process many requests with native provider batch APIs or concurrent fallback. OpenAI, Anthropic, and Google batches are processed server-side — OpenAI at 50% cost, Anthropic with async processing for up to 10K requests, Google at 50% cost via `batchGenerateContent`. Other providers fall back to concurrent execution automatically.

### Submit and wait

```python
results = await client.batches.create_and_poll({
    "model": "openai/gpt-4o-mini",
    "requests": [
        {"custom_id": "req-1", "messages": [{"role": "user", "content": "Summarize AI"}]},
        {"custom_id": "req-2", "messages": [{"role": "user", "content": "Summarize ML"}]},
    ],
})

for result in results["results"]:
    print(result["custom_id"], result["response"]["choices"][0]["message"]["content"])
```

### Submit now, check later

```python
# Submit and get the batch ID
batch = await client.batches.create({
    "model": "anthropic/claude-haiku-4-5",
    "requests": [
        {"custom_id": "req-1", "messages": [{"role": "user", "content": "Summarize AI"}]},
    ],
})
print(batch["id"])  # "batch-abc123"

# Check status any time
status = await client.batches.get("batch-abc123")
print(status["status"])  # "pending", "processing", "completed"

# Wait for results when ready
results = await client.batches.poll("batch-abc123")

# List all batches
all_batches = await client.batches.list()

# Cancel a batch
await client.batches.cancel("batch-abc123")
```

### Automatic max_tokens

When `max_tokens` isn't set on a batch request, anymodel automatically calculates a safe value per-request based on the estimated input size and the model's context window. This prevents truncated responses and context overflow errors without requiring you to hand-tune each request in a large batch.

### Batch configuration

```python
client = AnyModel({
    "batch": {
        "poll_interval": 10.0,          # default poll interval in seconds
        "concurrency_fallback": 10,      # concurrent request limit for non-native providers
    },
    "io": {
        "read_concurrency": 30,          # concurrent file reads (default: 20)
        "write_concurrency": 15,         # concurrent file writes (default: 10)
    },
})
```

## Configuration

```python
client = AnyModel({
    "anthropic": {"api_key": "sk-ant-..."},
    "openai": {"api_key": "sk-..."},
    "aliases": {
        "default": "anthropic/claude-sonnet-4-6",
        "fast": "anthropic/claude-haiku-4-5",
        "smart": "anthropic/claude-opus-4-6",
    },
    "defaults": {
        "temperature": 0.7,
        "max_tokens": 4096,
        "retries": 2,
        "timeout": 120,  # HTTP timeout in seconds (default: 120 = 2 min, flex: 600 = 10 min)
    },
})

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
    "api_key": "${ANTHROPIC_API_KEY}"
  },
  "aliases": {
    "default": "anthropic/claude-sonnet-4-6"
  },
  "defaults": {
    "temperature": 0.7,
    "max_tokens": 4096
  }
}
```

`${ENV_VAR}` references are interpolated from environment variables.

## Custom Providers

Add any OpenAI-compatible endpoint:

```python
client = AnyModel({
    "custom": {
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "models": ["llama3.3", "mistral"],
        },
    },
})

response = await client.chat.completions.create(
    model="ollama/llama3.3",
    messages=[{"role": "user", "content": "Hello from Ollama"}],
)
```

## Server Mode

Run as a standalone HTTP server compatible with the OpenAI SDK:

```bash
pip install anymodel[server]
anymodel serve --port 4141
```

Then point any OpenAI-compatible client at it:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:4141/api/v1", api_key="unused")
response = client.chat.completions.create(
    model="anthropic/claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Hello via server"}],
)
```

## Also Available

- **Node.js**: [`@probeo/anymodel`](https://github.com/probeo-io/anymodel) on npm
- **Go**: [`anymodel-go`](https://github.com/probeo-io/anymodel-go)

## License

MIT
