"""Shared test fixtures."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest


@pytest.fixture
def tmp_batch_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for batch tests."""
    batch_dir = tmp_path / "batches"
    batch_dir.mkdir()
    yield batch_dir
    shutil.rmtree(batch_dir, ignore_errors=True)
