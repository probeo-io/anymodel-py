"""Configuration resolution and merging."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


def _interpolate_env(value: Any) -> Any:
    """Replace ${ENV_VAR} references with environment variable values."""
    if isinstance(value, str):
        return re.sub(
            r"\$\{(\w+)\}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            value,
        )
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override into base. Override wins on conflicts."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_config_file(path: Path) -> dict[str, Any]:
    """Load and interpolate a config file."""
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return _interpolate_env(raw)
    except Exception:
        return {}


def resolve_config(programmatic: dict[str, Any] | None = None, cwd: str | None = None) -> dict[str, Any]:
    """Resolve configuration from all sources.

    Resolution order (highest to lowest priority):
    1. Programmatic options
    2. Local anymodel.config.json
    3. Global ~/.anymodel/config.json
    4. Environment variables
    """
    # Start with env var defaults
    config: dict[str, Any] = {}

    env_providers = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "groq": "GROQ_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "xai": "XAI_API_KEY",
        "together": "TOGETHER_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
        "perplexity": "PERPLEXITY_API_KEY",
    }

    for provider, env_var in env_providers.items():
        key = os.environ.get(env_var)
        if key:
            config[provider] = {"api_key": key}

    # Ollama (uses base URL, no API key)
    ollama_url = os.environ.get("OLLAMA_BASE_URL")
    if ollama_url:
        config["ollama"] = {"base_url": ollama_url}

    # Global config
    global_config = _load_config_file(Path.home() / ".anymodel" / "config.json")
    config = _deep_merge(config, global_config)

    # Local config
    work_dir = Path(cwd) if cwd else Path.cwd()
    local_config = _load_config_file(work_dir / "anymodel.config.json")
    config = _deep_merge(config, local_config)

    # Programmatic config (highest priority)
    if programmatic:
        config = _deep_merge(config, programmatic)

    return config
