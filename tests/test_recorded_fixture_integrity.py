"""Validate the real checked-in capture without loading the model/checkpoint."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fixture_compiler", ROOT / "scripts/compiler_spike.py")
spike = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(spike)


def test_committed_reference_bytes_and_tensor_metadata():
    base = ROOT / "tests/fixtures/cpu-reference"
    reference, _ = spike.read_reference(
        base / "reference.json",
        "d1cb5aaaebdc21b5b8c6db1287811fb098659370978e9d1f50e593b329acb085",
        ROOT / "configs/checkpoint-lock.json",
        ROOT / "configs/reference-cases.json",
    )
    calls = tensors = 0
    for case in reference["cases"]:
        for call in case["forward_calls"]:
            loaded = spike.load_call(base, case["id"], call)
            calls += 1
            tensors += len(loaded)
    assert len(reference["cases"]) == 6
    assert calls == 5
    assert tensors == 35
