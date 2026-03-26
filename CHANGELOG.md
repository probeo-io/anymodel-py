# Changelog

All notable changes to this project will be documented in this file.

## [0.5.1] - 2026-03-26

### Fixed

- Batch cost calculations now apply 50% discount for native batch APIs (OpenAI, Anthropic, Google)

## [0.5.0] - 2026-03-24

### Added

- **BatchBuilder API** — ergonomic batch construction with `client.batches.open(config)`
  - `add(prompt)` persists to disk immediately, caller just passes strings
  - `submit()` formats and dispatches to provider
  - `poll()` returns clean `succeeded`/`failed` results with per-item costs
  - `retry(failed)` creates a new builder pre-loaded with failed items
- Poll logging: `log_to_console` parameter and `ANYMODEL_BATCH_POLL_LOG` env var

## [0.4.0] - 2026-03-19

### Added

- Automatic per-request cost calculation from bundled pricing data (323 models)
- Pricing fetched from OpenRouter at build time — always current as of last publish
- `BatchUsageSummary.estimated_cost` now calculated automatically from token usage
- Exported `calculate_cost()`, `get_model_pricing()`, `PRICING_AS_OF`
- `batch_mode` option on batch create requests — set to `"concurrent"` to force individual requests (e.g. for flex pricing)
- `service_tier` support on batch requests (concurrent path only — native batch already discounted)

## [0.3.1] - 2026-03-17

### Changed

- Concurrent batch processing now streams requests from disk instead of holding all in memory — safe for 10K+ request batches
- Tasks created incrementally with concurrency gating instead of all-at-once, preventing memory spikes on large batches

## [0.3.0] - 2026-03-17

### Added

- Native batch API support for OpenAI (JSONL upload, 50% cost reduction, 24hr processing window)
- Native batch API support for Anthropic (Message Batches API, up to 10K requests)
- Native batch API support for Google Gemini via `batchGenerateContent` (50% cost reduction)
- `OpenAIBatchAdapter`, `AnthropicBatchAdapter`, and `GoogleBatchAdapter` classes implementing `BatchAdapter` protocol
- Automatic provider detection — native batch for OpenAI/Anthropic/Google, concurrent fallback for others
- Automatic `max_tokens` estimation for batch requests — when not explicitly set, calculates a safe value per-request based on estimated input token count and model context/completion limits
- `resolve_max_tokens()` and `estimate_token_count()` utilities in `anymodel.utils`
- OpenAI `service_tier` support — set `service_tier: "flex"` on requests for 50% cost reduction with flexible latency
- Configurable HTTP request timeout — 2 minutes default for normal requests, 10 minutes for flex (`service_tier: "flex"`) requests, both settable via `set_default_timeout()` and `set_flex_timeout()`

## [0.2.0] - 2026-03-16

### Added

- Native Perplexity provider with static model listing (sonar, sonar-pro, sonar-reasoning, sonar-reasoning-pro, sonar-deep-research, r1-1776)
- Citation passthrough in Perplexity responses
- Cross-language links in README (Node.js, Go)

### Changed

- Perplexity upgraded from generic OpenAI-compatible adapter to dedicated native provider

## [0.1.0] - 2026-03-15

### Added

- AnyModel SDK client with `chat.completions.create()`, `models.list()`, `generation.get()`
- Provider adapters for OpenAI, Anthropic, and Google/Gemini
- Built-in providers: Mistral, Groq, DeepSeek, xAI, Together, Fireworks, Perplexity, Ollama
- Custom provider support for any OpenAI-compatible endpoint
- Unified tool calling and structured output across all providers
- Fallback routing with `models` array and `route: "fallback"`
- Streaming support (SSE)
- Automatic retry with exponential backoff on 429/502/503
- Per-provider rate limit tracking
- Batch processing with concurrent fallback
- Config file support (`anymodel.config.json`, `~/.anymodel/config.json`)
- Environment variable interpolation in config
- HTTP server mode (`anymodel serve`)

