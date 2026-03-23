import json
from pathlib import Path

import pytest


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def sample_jsonl_data() -> list[dict]:
    return [
        {"ID": "123_456_000001", "text": "hello world"},
        {"ID": "789_012_000002", "text": "foo bar baz"},
    ]


@pytest.fixture
def sample_jsonl_file(tmp_path: Path, sample_jsonl_data: list[dict]) -> Path:
    fpath = tmp_path / "sample.jsonl"
    with open(fpath, "w") as f:
        for record in sample_jsonl_data:
            f.write(json.dumps(record) + "\n")
    return fpath
