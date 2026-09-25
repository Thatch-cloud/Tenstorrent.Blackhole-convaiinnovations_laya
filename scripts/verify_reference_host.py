"""Fresh independent CPU/FP32 capture and fixed-threshold reference comparison."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compiler_spike import INPUTS, OUTPUTS, checked_file, compare_arrays, read_reference
from laya_tt.integrity import LoadedStateIntegrityError
from laya_tt.reference import capture_reference, sha256_file

BASELINE_SHA256 = "d1cb5aaaebdc21b5b8c6db1287811fb098659370978e9d1f50e593b329acb085"
ATOL = RTOL = 1e-4


def _host_metadata():
    versions = {}
    for name in ("torch", "numpy", "transformers", "safetensors", "tokenizers", "huggingface_hub"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "processor": platform.processor(),
            "logical_cpus": os.cpu_count(), "versions": versions}


def _array(base, case_id, name, metadata):
    import numpy as np
    path = checked_file(Path(base) / case_id, metadata["file"], metadata["sha256"])
    array = np.load(path, allow_pickle=False)
    if list(array.shape) != metadata["shape"] or str(array.dtype) != metadata["dtype"]:
        raise ValueError("Captured array shape/dtype differs from index metadata")
    numpy_to_torch = {"int64": "torch.int64", "int32": "torch.int32", "bool": "torch.bool", "float32": "torch.float32"}
    if numpy_to_torch.get(str(array.dtype)) != metadata["torch_dtype"]:
        raise ValueError("Captured array Torch dtype differs from index metadata")
    if name in OUTPUTS and array.dtype != np.dtype("float32"):
        raise ValueError("Reference outputs must retain CPU float32")
    return array


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate decoded answer key")
        result[key] = value
    return result


def _answers(path):
    def nonfinite(_value):
        raise ValueError("Nonfinite decoded answer")
    value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_pairs, parse_constant=nonfinite)
    if not isinstance(value, list) or any(not isinstance(item, dict) or not isinstance(item.get("answers"), dict) for item in value):
        raise ValueError("Decoded answers must contain native per-state answer objects")
    return value


def _exact_values(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, dict):
        return set(actual) == set(expected) and all(_exact_values(actual[key], expected[key]) for key in actual)
    if isinstance(actual, list):
        return len(actual) == len(expected) and all(_exact_values(a, e) for a, e in zip(actual, expected))
    return actual == expected


def compare_captures(baseline, captured, manifest_path, cases_path, *, baseline_sha256=BASELINE_SHA256):
    """Hash-validate both captures before exact-input/numeric/semantic comparison.

    The digest argument supports explicit synthetic fixtures in tests. Production
    CLI never overrides its pinned committed baseline or fixed tolerance policy.
    """
    import numpy as np
    baseline, captured = Path(baseline), Path(captured)
    reference, _ = read_reference(baseline, baseline_sha256, manifest_path, cases_path)
    actual, _ = read_reference(captured, sha256_file(captured), manifest_path, cases_path)
    result = {"inputs_exact": True, "outputs_within_experiment_tolerance": True,
              "decoded_answers_exact": True, "question_order_exact": True, "cases": []}
    if [case["id"] for case in reference["cases"]] != [case["id"] for case in actual["cases"]]:
        raise ValueError("Capture case identity/order differs from baseline")
    for expected_case, actual_case in zip(reference["cases"], actual["cases"]):
        case_id = expected_case["id"]
        if case_id != expected_case["input"]["id"] or case_id != actual_case["input"]["id"]:
            raise ValueError("Capture case identity does not match its fixture")
        if len(expected_case["forward_calls"]) != len(actual_case["forward_calls"]):
            raise ValueError("Capture forward-call count differs from baseline")
        row = {"id": case_id, "forward_calls": []}
        for index, (expected_call, actual_call) in enumerate(zip(expected_case["forward_calls"], actual_case["forward_calls"])):
            comparison = {"index": index, "inputs": {}, "outputs": {}}
            for name in INPUTS + OUTPUTS:
                expected = _array(baseline.parent, case_id, name, expected_call[name])
                observed = _array(captured.parent, case_id, name, actual_call[name])
                if name in INPUTS:
                    exact = observed.dtype == expected.dtype and observed.shape == expected.shape and np.array_equal(observed, expected)
                    comparison["inputs"][name] = {"exact": bool(exact), "shape": list(observed.shape), "dtype": str(observed.dtype)}
                    result["inputs_exact"] &= bool(exact)
                else:
                    try:
                        metrics = compare_arrays(observed, expected, ATOL, RTOL)
                    except ValueError as exc:
                        metrics = {"passed": False, "error": str(exc), "atol": ATOL, "rtol": RTOL}
                    comparison["outputs"][name] = metrics
                    result["outputs_within_experiment_tolerance"] &= metrics["passed"]
            row["forward_calls"].append(comparison)
        expected_answers = _answers(checked_file(baseline.parent, expected_case["answers"]["file"], expected_case["answers"]["sha256"]))
        actual_answers = _answers(checked_file(captured.parent, actual_case["answers"]["file"], actual_case["answers"]["sha256"]))
        row["decoded_answers_exact"] = _exact_values(actual_answers, expected_answers)
        row["question_order_exact"] = ([list(item["answers"]) for item in actual_answers]
                                       == [list(item["answers"]) for item in expected_answers])
        result["decoded_answers_exact"] &= row["decoded_answers_exact"]
        result["question_order_exact"] &= row["question_order_exact"]
        result["cases"].append(row)
    result["passed"] = all(result[key] for key in ("inputs_exact", "outputs_within_experiment_tolerance", "decoded_answers_exact", "question_order_exact"))
    return result


def run_check(*, root, output, manifest_path=None, cases_path=None, baseline=None,
              baseline_sha256=BASELINE_SHA256, capture_fn=capture_reference):
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest_path = Path(manifest_path) if manifest_path else root / "configs/checkpoint-lock.json"
    cases_path = Path(cases_path) if cases_path else root / "configs/reference-cases.json"
    baseline = Path(baseline) if baseline else root / "tests/fixtures/cpu-reference/reference.json"
    # Even an existing empty directory is refused, so no previous report or partial
    # run is silently reused. The caller may create its parent for workflow logs.
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "FAILED", "passed": False,
        "physical_acceptance": False, "tt_execution": False, "source_precision": "float32",
        "comparison_policy": {"atol": ATOL, "rtol": RTOL, "inputs": "exact", "decoded_answers": "exact_with_question_order",
                              "scope": "independent CPU host experiment; not hardware promotion"},
        "host": _host_metadata(), "baseline_sha256": baseline_sha256, "capture_sha256": None,
        "script_sha256": sha256_file(__file__)}
    try:
        reference, manifest = read_reference(baseline, baseline_sha256, manifest_path, cases_path)
        report.update(manifest_sha256=sha256_file(manifest_path), cases_sha256=sha256_file(cases_path),
                      source_commit=manifest["upstream"]["commit"],
                      checkpoint_revision=manifest["checkpoint"]["revision"],
                      checkpoint_files=manifest["checkpoint"]["files"], baseline_runtime=reference["runtime"])
        source = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        report["port_source_commit"] = source.stdout.strip() if source.returncode == 0 else None
        capture_fn(manifest_path, cases_path, output / "capture", root)
        captured = output / "capture/reference.json"
        report["capture_sha256"] = sha256_file(captured)
        captured_report, _ = read_reference(captured, report["capture_sha256"], manifest_path, cases_path)
        report["capture_runtime"] = captured_report["runtime"]
        report["loaded_state_integrity"] = captured_report["loaded_state_integrity"]
        report["comparison"] = compare_captures(baseline, captured, manifest_path, cases_path, baseline_sha256=baseline_sha256)
        report["passed"] = report["comparison"]["passed"]
        report["status"] = "PASSED" if report["passed"] else "COMPARISON_FAILED"
    except LoadedStateIntegrityError as exc:
        report.update(status="LOADED_STATE_INTEGRITY_FAILED", loaded_state_integrity=exc.report,
                      error_type=type(exc).__name__, error=str(exc))
    except Exception as exc:
        report.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
    with (output / "host-reference-check.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else args.root / args.output
    try:
        report = run_check(root=args.root, output=output)
    except FileExistsError:
        print("Refusing to reuse an existing host-reference output directory", file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "report": str(output / "host-reference-check.json")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
