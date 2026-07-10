from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def tiny_flow_path() -> Path:
    return Path(__file__).parent / "data" / "tiny_flow.nc"
