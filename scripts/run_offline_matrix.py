"""Bounded offline compiler matrix; each shape gets a fresh process and cache."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

REFERENCE_SHA = "d1cb5aaaebdc21b5b8c6db1287811fb098659370978e9d1f50e593b329acb085"
DESCRIPTOR_SHA = "6028485af0f8c233185b87277f36061c2460db606f14a47bcb8284f5af5f56f3"
TOOLCHAIN = "873c53c4ff84bd91112a1f43e788cfed75aaa914"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def targets(reference, selected):
    available = {case["id"]: case for case in reference["cases"]}
    if len(available) != len(reference["cases"]) or len(set(selected)) != len(selected):
        raise ValueError("Duplicate case IDs")
    result = []
    for name in selected:
        if not re.fullmatch(r"[a-z0-9-]+", name) or name not in available:
            raise ValueError("Unknown or unsafe case ID")
        calls = available[name]["forward_calls"]
        if not calls:
            raise ValueError("Selected case has no tensor graph")
        result.extend((name, index) for index in range(len(calls)))
    return result


def inspect_report(directory, case, call, profile_bucket=None, precision="float32", policy_sha256=None):
    path = directory / "report.json"
    report = json.loads(path.read_text())
    evidence = {"report_sha256": sha(path), "status": report["status"], "artifacts": {}}
    if (report.get("mode") != "tt-compile-only" or report.get("compile_only") is not True
            or report.get("physical_acceptance") is not False or report.get("dtype") != precision
            or report.get("reference_sha256") != REFERENCE_SHA
            or any(key in report for key in ("graph_executed", "device_execution"))):
        raise ValueError("Report is not the pinned offline precision experiment")
    if report["status"] != "COMPILED":
        return evidence
    if precision == "mixed-bf16-fp32":
        policy = report.get("precision_policy", {})
        if (not policy_sha256 or policy.get("name") != precision
                or policy.get("helper_sha256") != policy_sha256
                or report.get("candidate_loaded_state_integrity", {}).get("verified") is not True):
            raise ValueError("Mixed precision policy or converted-state integrity mismatch")
        inventory = policy.get("inventory", {})
        parameters = inventory.get("parameters", {})
        if not parameters or not inventory.get("buffers"):
            raise ValueError("Missing mixed precision tensor inventory")
        for name, item in parameters.items():
            if name.startswith("act_head."):
                expected = "torch.float32"
            elif name.startswith(("encoder.", "head.", "type_emb.", "scorer.")):
                expected = "torch.bfloat16"
            else:
                raise ValueError("Unknown mixed precision parameter")
            if item.get("dtype") != expected:
                raise ValueError("Mixed precision parameter dtype mismatch")
    compiled = report["compilation"]
    if compiled.get("profile_bucket") != profile_bucket:
        raise ValueError("Compiled profile identity mismatch")
    if profile_bucket is not None and compiled.get("input_shapes") != {
            "input_ids": [1, profile_bucket], "attention_mask": [1, profile_bucket],
            "marker_pos": [1, 64], "marker_mask": [1, 64], "qtype": [1]}:
        raise ValueError("Compiled profile shapes mismatch")
    if (compiled["case"] != case or compiled["call"] != call
            or compiled.get("numerical_comparison_performed") is not False
            or report.get("loaded_state_integrity", {}).get("verified") is not True):
        raise ValueError("Compiled report identity or integrity mismatch")
    preflight = report["compile_only_preflight"]
    if (preflight.get("device_nodes_exposed") is not False
            or preflight["compiler"]["commit"] != TOOLCHAIN
            or preflight["system_descriptor"]["sha256"] != DESCRIPTOR_SHA):
        raise ValueError("Compiler or descriptor provenance mismatch")
    expected_names = {f"{case}-{call:03d}{suffix}" for suffix in (".ttnn", "_ttir.mlir", "_ttnn.mlir")}
    if set(compiled["artifacts"]) != expected_names:
        raise ValueError("Compiled artifact set mismatch")
    for name, item in compiled["artifacts"].items():
        artifact = (directory / item["file"]).resolve()
        if not artifact.is_relative_to(directory.resolve()) or not artifact.is_file():
            raise ValueError("Unsafe or missing compiled artifact")
        if artifact.stat().st_size != item["bytes"] or item["bytes"] == 0 or sha(artifact) != item["sha256"]:
            raise ValueError("Compiled artifact hash/size mismatch")
        evidence["artifacts"][name] = item
    evidence["input_shapes"] = compiled["input_shapes"]
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", nargs="+", default=["mixed-question-widths", "mixed-state-lengths", "conversation-truncation", "option-temperature-buckets"])
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--profile-buckets", nargs="+", type=int, choices=(32, 64, 128, 256, 512))
    parser.add_argument("--precision", choices=("float32", "mixed-bf16-fp32"), default="float32")
    args = parser.parse_args(argv)
    if platform.system() != "Linux" or list(Path("/dev/tenstorrent").glob("*")):
        raise RuntimeError("Offline matrix requires Linux without TT device nodes")
    if os.environ.get("TT_VISIBLE_DEVICES") or "TT_COMPILE_ONLY_SYSTEM_DESC" in os.environ:
        raise RuntimeError("Clear inherited accelerator selectors before matrix launch")
    if not 1 <= args.timeout <= 3600:
        raise ValueError("Timeout must be 1..3600 seconds per graph")
    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    if not output.is_relative_to(root) or output.exists():
        raise ValueError("Output must be a new directory under the repository")
    ref = root / "tests/fixtures/cpu-reference/reference.json"
    if sha(ref) != REFERENCE_SHA:
        raise ValueError("Reference report hash mismatch")
    jobs = targets(json.loads(ref.read_text()), args.cases)
    if args.profile_buckets:
        if args.cases != ["single-option"] or len(set(args.profile_buckets)) != len(args.profile_buckets):
            raise ValueError("Profile matrix requires single-option and unique buckets")
        jobs = [(case, call, bucket) for case, call in jobs for bucket in args.profile_buckets]
    else:
        jobs = [(case, call, None) for case, call in jobs]
    output.mkdir(parents=True)
    policy_sha256 = sha(root / "scripts/probe_bf16_cpu.py") if args.precision == "mixed-bf16-fp32" else None
    aggregate = {"schema_version": 1, "kind": "offline_fp32_compiler_matrix" if args.precision == "float32" else "offline_mixed_precision_compiler_matrix",
                 "physical_acceptance": False, "numerical_comparison_performed": False,
                 "reference_sha256": REFERENCE_SHA, "descriptor_sha256": DESCRIPTOR_SHA,
                 "toolchain_commit": TOOLCHAIN, "dtype": args.precision, "cases": [],
                 "precision_policy_sha256": policy_sha256,
                 "compiler_harness_sha256": sha(root / "scripts/compiler_spike.py"),
                 "profile_helper_sha256": sha(root / "scripts/shape_profiles.py") if args.profile_buckets else None,
                 "runner_sha256": sha(Path(__file__)),
                 "note": "Elapsed times are host compilation duration, not inference latency"}
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", GIT_CONFIG_COUNT="2", GIT_CONFIG_KEY_0="safe.directory",
               GIT_CONFIG_VALUE_0=str(root / ".cache/upstream/laya"),
               GIT_CONFIG_KEY_1="core.autocrlf", GIT_CONFIG_VALUE_1="true")
    for case, call, bucket in jobs:
        name = f"{case}-{call:03d}"
        if bucket is not None:
            name += f"-profile-{bucket}"
        destination = output / name
        log = output / f"{name}.log"
        command = [sys.executable, str(root / "scripts/compiler_spike.py"), "--mode", "tt-compile-only",
                   "--toolchain-commit", TOOLCHAIN, "--reference", str(ref), "--reference-sha256", REFERENCE_SHA,
                   "--system-desc", str(root / "configs/compiler/migrated-p150/p150-migrated-nightly.ttsys"),
                   "--system-desc-sha256", DESCRIPTOR_SHA, "--case-id", case, "--call-index", str(call),
                   "--output", str(destination), "--precision", args.precision]
        if bucket is not None:
            command.extend(["--profile-bucket", str(bucket)])
        started = time.monotonic()
        row = {"case": case, "call": call, "profile_bucket": bucket, "directory": name, "log": log.name}
        with log.open("wb") as stream:
            process = subprocess.Popen(command, cwd=root, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                row["exit_code"] = process.wait(timeout=args.timeout)
                row.update(inspect_report(destination, case, call, bucket, args.precision, policy_sha256))
                if row["exit_code"] != 0 and row["status"] == "COMPILED":
                    raise ValueError("Compiled report conflicts with failed process")
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                row["status"] = "TIMEOUT"
            except Exception as exc:
                row.update(status="INVALID_EVIDENCE", reason=str(exc))
        row.update(elapsed_seconds=time.monotonic() - started, log_sha256=sha(log))
        aggregate["cases"].append(row)
        aggregate["status"] = "COMPILED" if len(aggregate["cases"]) == len(jobs) and all(item["status"] == "COMPILED" for item in aggregate["cases"]) else "INCOMPLETE_OR_FAILED"
        (output / "matrix.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
        print(json.dumps(row), flush=True)
    return 0 if aggregate["status"] == "COMPILED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
