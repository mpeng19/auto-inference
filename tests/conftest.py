import json
import pathlib

import pytest

DATA = pathlib.Path(__file__).parent / "data"


@pytest.fixture
def sweep() -> dict:
    """The real 1xH100 baseline sweep (run 1788287578), trimmed to what the
    product reads. Every number the product claims is reproduced from this."""
    return json.loads((DATA / "sweep-1xh100.json").read_text())


@pytest.fixture
def root(tmp_path) -> pathlib.Path:
    d = tmp_path / "run"
    d.mkdir()
    return d
