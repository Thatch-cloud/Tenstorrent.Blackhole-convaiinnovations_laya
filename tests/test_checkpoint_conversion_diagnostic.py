import importlib.util
import io
import json
from pathlib import Path
import struct
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/diagnose_checkpoint_conversion.py"
SPEC = importlib.util.spec_from_file_location("diagnostic", PATH)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def test_scalar_oracle_identifies_observed_wrong_bits():
    np = pytest.importorskip("numpy")
    source = np.array([[1, -0.7265625]], dtype=np.float16)
    expected = source.astype(np.float32)
    actual = expected.copy()
    actual[0, 1] = -0.002838134765625
    stream = io.BytesIO(b"prefix" + source.tobytes())
    result = diagnostic.compare_bits(expected, actual, source, stream, 6)
    first = result["first_mismatch"]
    assert first["index"] == [0, 1]
    assert first["scalar_oracle"]["checkpoint_byte_offset"] == 8
    assert first["scalar_oracle"]["numpy_expected_matches_scalar"] is True
    assert first["scalar_oracle"]["observed_matches_scalar"] is False
    assert first["scalar_oracle"]["independent_fp32_bits"] == first["expected_bits"]
    assert result["mismatched_elements"] == 1


def test_scalar_oracle_can_identify_wrong_numpy_expectation():
    np = pytest.importorskip("numpy")
    source = np.array([0.25], dtype=np.float16)
    wrong_expected = np.array([0.5], dtype=np.float32)
    actual = np.array([0.25], dtype=np.float32)
    result = diagnostic.compare_bits(wrong_expected, actual, source, io.BytesIO(source.tobytes()))
    oracle = result["first_mismatch"]["scalar_oracle"]
    assert oracle["numpy_expected_matches_scalar"] is False
    assert oracle["observed_matches_scalar"] is True


def test_bit_comparison_detects_signed_zero():
    np = pytest.importorskip("numpy")
    source = np.array([-0.0], dtype=np.float16)
    expected = source.astype(np.float32)
    actual = np.array([0.0], dtype=np.float32)
    assert diagnostic.compare_bits(expected, actual, source)["mismatched_elements"] == 1


def test_header_rejects_overlapping_ranges():
    header = {"one": {"dtype": "F16", "shape": [2], "data_offsets": [0, 4]},
              "two": {"dtype": "F16", "shape": [2], "data_offsets": [2, 6]}}
    encoded = json.dumps(header).encode()
    payload = struct.pack("<Q", len(encoded)) + encoded + bytes(6)
    with pytest.raises(ValueError, match="Overlapping"):
        diagnostic.read_header(io.BytesIO(payload), len(payload))


def test_small_real_checkpoint_covers_each_path(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    checkpoint = tmp_path / "test.safetensors"
    safetensors_torch.save_file({"weights": torch.tensor([1, -0.7265625], dtype=torch.float16),
                               "temperature": torch.tensor([1], dtype=torch.float32)}, str(checkpoint))
    monkeypatch.setattr(diagnostic, "CHECKPOINT_SHA256", diagnostic.file_sha(checkpoint))
    monkeypatch.setattr(diagnostic, "EXPECTED_TENSORS", 2)
    monkeypatch.setattr(diagnostic, "EXPECTED_ELEMENTS", 3)
    assert diagnostic.diagnose(checkpoint, tmp_path / "out") == 0
    report = json.loads((tmp_path / "out/report.json").read_text())
    assert report["status"] == "MATCHED"
    assert report["checkpoint_sha256_before"] == report["checkpoint_sha256_after"]
    assert all(len(row["checks"]) == 5 for row in report["tensors"])
    assert report["model_repaired"] is False
