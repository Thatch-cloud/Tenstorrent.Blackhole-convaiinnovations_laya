"""Dataset integrity must fail closed before producing evaluation inputs."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "quality_preparer", Path(__file__).resolve().parents[1] / "scripts/prepare_quality_suite.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("failure", ["hash", "rows"])
def test_corrupt_source_does_not_produce_cases(tmp_path, failure):
    data = b'{"text":"A synthetic review.","label":2}\n'
    (tmp_path / "sst5.jsonl").write_bytes(data)
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"datasets": [{
        "id": "sst5", "file": "test.jsonl",
        "sha256": "0" * 64 if failure == "hash" else module.digest(data),
        "rows": 1 if failure == "hash" else 2,
    }]}))
    output = tmp_path / "cases.jsonl"
    with pytest.raises(ValueError, match="mismatch"):
        module.prepare(lock, tmp_path, output)
    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()
