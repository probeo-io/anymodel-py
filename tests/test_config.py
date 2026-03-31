"""Tests for config resolution and merging."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from anymodel._config import resolve_config


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for config tests."""
    return tmp_path


def test_returns_programmatic_config_when_no_files(config_dir: Path) -> None:
    config = resolve_config({"anthropic": {"api_key": "sk-test"}}, str(config_dir))
    assert config["anthropic"]["api_key"] == "sk-test"


def test_loads_local_config_file(config_dir: Path) -> None:
    (config_dir / "anymodel.config.json").write_text(
        json.dumps({"aliases": {"fast": "anthropic/claude-haiku-4-5"}}),
    )
    config = resolve_config({}, str(config_dir))
    assert config["aliases"]["fast"] == "anthropic/claude-haiku-4-5"


def test_programmatic_overrides_local_config(config_dir: Path) -> None:
    (config_dir / "anymodel.config.json").write_text(
        json.dumps({"defaults": {"temperature": 0.5}}),
    )
    config = resolve_config({"defaults": {"temperature": 0.9}}, str(config_dir))
    assert config["defaults"]["temperature"] == 0.9


def test_deep_merges_provider_configs(config_dir: Path) -> None:
    (config_dir / "anymodel.config.json").write_text(
        json.dumps({"anthropic": {"default_model": "claude-haiku-4-5"}}),
    )
    config = resolve_config({"anthropic": {"api_key": "sk-test"}}, str(config_dir))
    assert config["anthropic"]["api_key"] == "sk-test"
    assert config["anthropic"]["default_model"] == "claude-haiku-4-5"


def test_interpolates_env_var_in_config(config_dir: Path) -> None:
    os.environ["__TEST_KEY"] = "sk-from-env"
    try:
        (config_dir / "anymodel.config.json").write_text(
            json.dumps({"openai": {"api_key": "${__TEST_KEY}"}}),
        )
        config = resolve_config({}, str(config_dir))
        assert config["openai"]["api_key"] == "sk-from-env"
    finally:
        del os.environ["__TEST_KEY"]


def test_picks_up_api_keys_from_env_vars(config_dir: Path) -> None:
    orig = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "sk-env-anthropic"
    try:
        config = resolve_config({}, str(config_dir))
        assert config["anthropic"]["api_key"] == "sk-env-anthropic"
    finally:
        if orig is not None:
            os.environ["ANTHROPIC_API_KEY"] = orig
        else:
            del os.environ["ANTHROPIC_API_KEY"]


def test_handles_missing_config_files_gracefully() -> None:
    config = resolve_config({}, "/nonexistent/path")
    assert isinstance(config, dict)
