# anymodel-py

Unified Python SDK for routing LLM requests across OpenAI, Anthropic, Google, and other providers. OpenRouter-compatible API with built-in batch processing, fallback routing, and streaming.

Published as `anymodel-py` on PyPI; imported as `anymodel` in Python.

## Build & Test

```bash
pip install -e ".[dev]"   # editable install with dev deps
pytest                    # run tests (asyncio_mode = auto)
ruff check src tests      # lint
ruff format src tests     # format
mypy src                  # type check (strict mode)
```

Optional server dependencies:

```bash
pip install -e ".[server]"   # adds starlette + uvicorn
anymodel serve               # start HTTP server on :4141
```

## Architecture

- `src/anymodel/__init__.py` — Public API surface; re-exports `AnyModel` client and all types
- `src/anymodel/_client.py` — `AnyModel` class: main entry point, registers providers, exposes `chat.completions.create()`, `models.list()`, `batches.*`, and `generation.*` namespaces
- `src/anymodel/_router.py` — `Router`: resolves model strings, applies defaults/transforms, handles fallback routing across providers, records generation stats
- `src/anymodel/_types.py` — All TypedDict types mirroring the OpenAI/OpenRouter API surface (requests, responses, config, errors)
- `src/anymodel/_config.py` — Config resolution: env vars < `~/.anymodel/config.json` < local `anymodel.config.json` < programmatic config
- `src/anymodel/_server.py` — Optional Starlette ASGI app exposing the router as an HTTP API
- `src/anymodel/_cli.py` — CLI entry point (`anymodel serve`)
- `src/anymodel/providers/` — Provider adapters:
  - `_adapter.py` — `ProviderAdapter` and `BatchAdapter` protocols (interfaces)
  - `_registry.py` — `ProviderRegistry`: stores and retrieves adapters by slug
  - `_openai.py` — OpenAI adapter (also the template for OpenAI-compatible providers)
  - `_anthropic.py` — Anthropic adapter (messages API translation)
  - `_google.py` — Google Gemini adapter
  - `_custom.py` — Generic adapter for any OpenAI-compatible endpoint
- `src/anymodel/batch/` — Batch processing:
  - `_manager.py` — `BatchManager`: native provider batches or concurrent-request fallback
  - `_store.py` — File-based batch state persistence
- `src/anymodel/utils/` — Shared utilities:
  - `_model_parser.py` — Parses `provider/model` strings and resolves aliases
  - `_validate.py` — Request validation
  - `_retry.py` — Async retry wrapper
  - `_rate_limiter.py` — Per-provider rate-limit tracking
  - `_transforms.py` — Message transforms (e.g., middle-out)
  - `_id.py` — ID generation (`gen-*`, `batch-*`)
  - `_fs_io.py` — Async file I/O with configurable concurrency
  - `_generation_stats.py` — In-memory generation stats store
- `tests/` — pytest tests; async tests run automatically via `asyncio_mode = auto`

## Key Design Decisions

- **OpenAI-compatible API surface** — `client.chat.completions.create()` mirrors the OpenAI SDK so switching is low-friction
- **Model strings use `provider/model` format** — e.g., `openai/gpt-4o`, `anthropic/claude-sonnet-4-20250514`; the router parses and dispatches
- **Provider adapters are protocol-based** — `ProviderAdapter` is a `typing.Protocol`, not an ABC; adapters satisfy it structurally
- **All network I/O uses httpx** — no provider-specific SDKs; adapters translate between provider APIs and the unified format directly
- **Batch processing has two paths** — native (provider batch API) and concurrent (semaphore-bounded parallel requests); the manager picks automatically
- **Config merges four layers** — env vars, global config, local config, programmatic; each layer deep-merges over the previous
- **Responses are re-prefixed** — IDs become `gen-*`, model names become `provider/model` in responses for consistency
- **All modules are private** (`_module.py`) — public API is exclusively through `__init__.py` exports
- **Python 3.10+** required; type annotations use `dict`, `list`, `X | Y` syntax (no `typing.Dict` / `Optional`)

## Adding a New Provider

1. Create `src/anymodel/providers/_yourprovider.py` implementing the `ProviderAdapter` protocol:
   - `name` property returning the provider slug
   - `send_request()` — translate request to provider format, call API via httpx, translate response back
   - `send_streaming_request()` — same but return an `AsyncIterator` of SSE chunks
   - `list_models()` — fetch and normalize available models
   - `supports_parameter(param)` — return whether the provider supports a given parameter
   - `supports_batch()` — return `True` if native batch is available
   - `translate_error()` — normalize provider errors to `AnyModelError`
2. Add a `create_yourprovider_adapter(api_key)` factory function.
3. Register in `_client.py` `_register_providers()` — read the API key from config or env var, call `self._registry.register("slug", adapter)`.
4. If the provider has a native batch API, also implement the `BatchAdapter` protocol and register it on the `BatchManager`.
5. Add tests in `tests/providers/` using `respx` to mock HTTP responses.
