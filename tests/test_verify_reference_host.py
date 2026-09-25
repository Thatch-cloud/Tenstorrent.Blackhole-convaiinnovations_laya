import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("verify_reference_host", ROOT / "scripts/verify_reference_host.py")
host_check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(host_check)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return host_check.sha256_file(path)


def fixture(tmp_path):
    manifest = {"schema_version": 1, "upstream": {"commit": "a"*40},
                "checkpoint": {"revision": "b"*40, "files": {"model.safetensors": "c"*64}}}
    case = {"id": "tiny", "state": "fixture", "questions": {
        "first": {"type": "noul", "instructions": "first"},
        "second": {"type": "noul", "instructions": "second"}}}
    manifest_path, cases_path = tmp_path / "manifest.json", tmp_path / "cases.json"
    manifest_sha = write_json(manifest_path, manifest)
    cases_sha = write_json(cases_path, {"schema_version": 1, "cases": [case]})
    baseline = tmp_path / "baseline"
    (baseline / "tiny").mkdir(parents=True)
    calls = {}
    for name in host_check.INPUTS + host_check.OUTPUTS:
        dtype = np.float32 if name in host_check.OUTPUTS else (np.bool_ if name == "marker_mask" else np.int64)
        values = np.asarray([[.1, .2]] if name in host_check.OUTPUTS else [[1, 1]], dtype=dtype)
        path = baseline / "tiny" / (name + ".npy")
        np.save(path, values, allow_pickle=False)
        calls[name] = {"file": path.name, "sha256": host_check.sha256_file(path),
                       "shape": list(values.shape), "dtype": str(values.dtype),
                       "torch_dtype": "torch." + str(values.dtype)}
    answers = [{"model": "laya-rl-agent", "answers": {"first": {"noul": .5}, "second": {"noul": .8}},
                "usage": {"input_tokens": 2, "output_tokens": 0}}]
    answer_sha = write_json(baseline / "tiny/answers.json", answers)
    report = {"schema_version": 1, "manifest": manifest, "manifest_sha256": manifest_sha,
        "cases_sha256": cases_sha, "loaded_state_integrity": {"verified": True},
        "runtime": {"device": "cpu", "weight_dtype": "torch.float32", "autocast": False},
        "cases": [{"id": "tiny", "input": case, "forward_calls": [calls],
                   "answers": {"file": "tiny/answers.json", "sha256": answer_sha}}]}
    digest = write_json(baseline / "reference.json", report)
    captured = tmp_path / "captured"
    shutil.copytree(baseline, captured)
    return baseline / "reference.json", captured / "reference.json", manifest_path, cases_path, digest


def compare(paths):
    baseline, captured, manifest, cases, digest = paths
    return host_check.compare_captures(baseline, captured, manifest, cases, baseline_sha256=digest)


def replace_tensor(captured, name, delta):
    report = json.loads(captured.read_text())
    metadata = report["cases"][0]["forward_calls"][0][name]
    path = captured.parent / "tiny" / metadata["file"]
    value = np.load(path, allow_pickle=False)
    value[0, 0] += delta
    np.save(path, value, allow_pickle=False)
    metadata["sha256"] = host_check.sha256_file(path)
    write_json(captured, report)


def replace_answers(captured, change):
    report = json.loads(captured.read_text())
    path = captured.parent / "tiny/answers.json"
    answers = json.loads(path.read_text())
    change(answers)
    report["cases"][0]["answers"]["sha256"] = write_json(path, answers)
    write_json(captured, report)


def test_identical_tiny_captures_report_all_comparisons(tmp_path):
    result = compare(fixture(tmp_path))
    assert result["passed"]
    row = result["cases"][0]["forward_calls"][0]
    assert set(row["inputs"]) == set(host_check.INPUTS)
    assert set(row["outputs"]) == set(host_check.OUTPUTS)
    assert row["outputs"]["logits"]["max_abs_error"] == 0
    assert row["outputs"]["act_logits"]["atol"] == 1e-4


def test_tampered_tensor_rejected_before_numeric_comparison(tmp_path):
    paths = fixture(tmp_path)
    (paths[1].parent / "tiny/logits.npy").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        compare(paths)


def test_exact_input_mismatch_fails_even_when_outputs_equal(tmp_path):
    paths = fixture(tmp_path)
    replace_tensor(paths[1], "input_ids", 1)
    result = compare(paths)
    assert not result["passed"] and not result["inputs_exact"]
    assert result["outputs_within_experiment_tolerance"]


def test_numeric_output_error_fails_predeclared_tolerance(tmp_path):
    paths = fixture(tmp_path)
    replace_tensor(paths[1], "act_logits", .02)
    result = compare(paths)
    assert not result["passed"]
    assert result["inputs_exact"] and result["decoded_answers_exact"]
    assert not result["outputs_within_experiment_tolerance"]
    assert result["cases"][0]["forward_calls"][0]["outputs"]["act_logits"]["max_abs_error"] > .01


def test_answer_order_drift_cannot_be_hidden_by_equal_dict_values(tmp_path):
    paths = fixture(tmp_path)
    def reverse(answers):
        answers[0]["answers"] = dict(reversed(list(answers[0]["answers"].items())))
    replace_answers(paths[1], reverse)
    result = compare(paths)
    assert result["decoded_answers_exact"]
    assert not result["question_order_exact"] and not result["passed"]


def test_semantic_answer_drift_fails_even_when_logits_within_tolerance(tmp_path):
    paths = fixture(tmp_path)
    replace_tensor(paths[1], "logits", 1e-6)
    replace_answers(paths[1], lambda answers: answers[0]["answers"]["first"].update(noul=.5001))
    result = compare(paths)
    assert result["outputs_within_experiment_tolerance"]
    assert not result["decoded_answers_exact"] and not result["passed"]


def test_array_metadata_mismatch_is_rejected(tmp_path):
    paths = fixture(tmp_path)
    report = json.loads(paths[1].read_text())
    report["cases"][0]["forward_calls"][0]["qtype"]["shape"] = [999]
    write_json(paths[1], report)
    with pytest.raises(ValueError, match="metadata"):
        compare(paths)


def test_loaded_state_failure_retains_structured_report_without_retry(tmp_path):
    baseline, _, manifest, cases, digest = fixture(tmp_path)
    details = {"verified": False, "mismatch_tensors": 1,
               "mismatches": [{"key": "encoder.weight", "reason": "value_mismatch", "different_elements": 2}]}
    called = []
    def fail(*args):
        called.append(args)
        raise host_check.LoadedStateIntegrityError(details)
    output = tmp_path / "fresh-run"
    report = host_check.run_check(root=tmp_path, output=output, manifest_path=manifest, cases_path=cases,
                                 baseline=baseline, baseline_sha256=digest, capture_fn=fail)
    stored = json.loads((output / "host-reference-check.json").read_text())
    assert len(called) == 1
    assert report == stored
    assert report["status"] == "LOADED_STATE_INTEGRITY_FAILED"
    assert report["loaded_state_integrity"] == details
    assert not report["passed"] and not report["physical_acceptance"] and not report["tt_execution"]
    assert report["source_precision"] == "float32"
    assert report["host"]["versions"]


def test_fresh_capture_success_and_no_overwrite(tmp_path):
    baseline, captured, manifest, cases, digest = fixture(tmp_path)
    calls = []
    def capture(_manifest, _cases, output, root):
        calls.append(output)
        assert not output.exists()
        shutil.copytree(captured.parent, output)
    output = tmp_path / "run"
    arguments = dict(root=tmp_path, output=output, manifest_path=manifest, cases_path=cases,
                     baseline=baseline, baseline_sha256=digest, capture_fn=capture)
    report = host_check.run_check(**arguments)
    assert report["status"] == "PASSED" and report["passed"]
    assert report["capture_sha256"] == digest
    prior = (output / "host-reference-check.json").read_bytes()
    with pytest.raises(FileExistsError):
        host_check.run_check(**arguments)
    assert len(calls) == 1
    assert (output / "host-reference-check.json").read_bytes() == prior


def test_baseline_tamper_stops_before_capture_and_preserves_failure(tmp_path):
    baseline, _, manifest, cases, digest = fixture(tmp_path)
    baseline.write_text("{}")
    called = []
    report = host_check.run_check(root=tmp_path, output=tmp_path / "run", manifest_path=manifest,
        cases_path=cases, baseline=baseline, baseline_sha256=digest, capture_fn=lambda *args: called.append(True))
    assert report["status"] == "FAILED" and not report["passed"]
    assert "index hash" in report["error"] and called == []


def test_cli_fails_for_failed_report_without_claiming_hardware(monkeypatch, tmp_path):
    monkeypatch.setattr(host_check, "run_check", lambda **kwargs: {"status": "COMPARISON_FAILED", "passed": False})
    assert host_check.main(["--root", str(tmp_path), "--output", "run"]) == 1
