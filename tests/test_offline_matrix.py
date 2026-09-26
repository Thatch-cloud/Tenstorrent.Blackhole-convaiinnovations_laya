import importlib.util
import json
from pathlib import Path
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/run_offline_matrix.py"
SPEC = importlib.util.spec_from_file_location("matrix", PATH)
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


def test_matrix_selects_each_forward_and_rejects_unsafe_or_empty():
    reference = {"cases": [{"id": "case", "forward_calls": [{}, {}]}, {"id": "empty", "forward_calls": []}]}
    assert matrix.targets(reference, ["case"]) == [("case", 0), ("case", 1)]
    for names in (["missing"], ["../case"], ["empty"], ["case", "case"]):
        with pytest.raises(ValueError):
            matrix.targets(reference, names)


def report_fixture(directory):
    artifacts = {}
    for suffix in (".ttnn", "_ttir.mlir", "_ttnn.mlir"):
        name = "case-000" + suffix
        path = directory / name
        path.write_bytes(b"unit fixture")
        artifacts[name] = {"file": name, "bytes": path.stat().st_size, "sha256": matrix.sha(path)}
    report = {"mode": "tt-compile-only", "compile_only": True, "physical_acceptance": False,
              "dtype": "float32", "reference_sha256": matrix.REFERENCE_SHA, "status": "COMPILED",
              "loaded_state_integrity": {"verified": True}, "compile_only_preflight": {
                  "device_nodes_exposed": False, "compiler": {"commit": matrix.TOOLCHAIN},
                  "system_descriptor": {"sha256": matrix.DESCRIPTOR_SHA}},
              "compilation": {"case": "case", "call": 0, "numerical_comparison_performed": False,
                              "artifacts": artifacts, "input_shapes": {}}}
    (directory / "report.json").write_text(json.dumps(report))
    return report


def test_artifact_hashes_verified(tmp_path):
    report_fixture(tmp_path)
    assert matrix.inspect_report(tmp_path, "case", 0)["status"] == "COMPILED"
    (tmp_path / "case-000.ttnn").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="hash/size"):
        matrix.inspect_report(tmp_path, "case", 0)


def test_profile_identity_and_all_input_shapes_must_match(tmp_path):
    report = report_fixture(tmp_path)
    compiled = report["compilation"]
    compiled["profile_bucket"] = 256
    compiled["input_shapes"] = {"input_ids": [1, 256], "attention_mask": [1, 256],
                                "marker_pos": [1, 64], "marker_mask": [1, 64], "qtype": [1]}
    (tmp_path / "report.json").write_text(json.dumps(report))
    assert matrix.inspect_report(tmp_path, "case", 0, 256)["status"] == "COMPILED"
    with pytest.raises(ValueError, match="profile identity"):
        matrix.inspect_report(tmp_path, "case", 0)
    compiled["input_shapes"]["marker_mask"] = [1, 1]
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="profile shapes"):
        matrix.inspect_report(tmp_path, "case", 0, 256)


@pytest.mark.parametrize("field,value", [("physical_acceptance", True), ("dtype", "bfloat16"),
                                        ("graph_executed", False), ("device_execution", True)])
def test_execution_or_wrong_dtype_report_rejected(tmp_path, field, value):
    report = report_fixture(tmp_path)
    report[field] = value
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        matrix.inspect_report(tmp_path, "case", 0)


def test_wrong_case_and_incomplete_artifacts_rejected(tmp_path):
    report = report_fixture(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        matrix.inspect_report(tmp_path, "other", 0)
    del report["compilation"]["artifacts"]["case-000.ttnn"]
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="artifact set"):
        matrix.inspect_report(tmp_path, "case", 0)
