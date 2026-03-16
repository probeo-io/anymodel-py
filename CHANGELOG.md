# Changelog

All notable changes to this project will be documented in this file.

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
- Batch processing with native OpenAI/Anthropic APIs and concurrent fallback
- Config file support (`anymodel.config.json`, `~/.anymodel/config.json`)
- Environment variable interpolation in config
- HTTP server mode (`anymodel serve`)

