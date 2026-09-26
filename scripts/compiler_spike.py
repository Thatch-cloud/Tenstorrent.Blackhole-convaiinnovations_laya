"""Pinned full-graph compiler experiment; CPU export is not TT acceptance."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import importlib.metadata
import re
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from laya_tt.reference import checkpoint_files, sha256_file, validate_manifest
from laya_tt.integrity import LoadedStateIntegrityError

INPUTS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
OUTPUTS = ("logits", "act_logits")
TT_XLA_AUDITED_COMMIT = "3bf6e4201f008fdf5a0ce5d244bc83ad155ee328"
OFFLINE_NIGHTLY_COMMIT = "873c53c4ff84bd91112a1f43e788cfed75aaa914"
OFFLINE_NIGHTLY_VERSION = "1.5.0.dev20260831000501"
P150_DESCRIPTOR_SHA256 = "655fabe5445624b0c9a6a312b90fb5e9a30a78614e43d331e9d622561e09b5ac"
MIGRATED_P150_SHA256 = "6028485af0f8c233185b87277f36061c2460db606f14a47bcb8284f5af5f56f3"
MIGRATED_P150_PROVENANCE_SHA256 = "94efbbb967aaabde069c779c56fa94758c8465d30a198b946a846ea2bb6ceb2c"

class Blocked(RuntimeError):
    pass

def checked_file(base, name, expected):
    base = Path(base).resolve()
    path = (base / name).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError(f"Artifact path escapes root or is missing: {name}")
    if sha256_file(path) != expected:
        raise ValueError(f"Artifact hash mismatch: {name}")
    return path

def read_reference(reference, digest, manifest_path, cases_path):
    reference = Path(reference)
    if sha256_file(reference) != digest:
        raise ValueError("Reference index hash mismatch")
    report = json.loads(reference.read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if report["schema_version"] != 1 or report["manifest"] != manifest:
        raise ValueError("Reference manifest identity mismatch")
    if sha256_file(manifest_path) != report["manifest_sha256"]:
        raise ValueError("Reference manifest bytes mismatch")
    if sha256_file(cases_path) != report["cases_sha256"]:
        raise ValueError("Reference fixture definition hash mismatch")
    definitions = json.loads(Path(cases_path).read_text(encoding="utf-8"))["cases"]
    if [case["input"] for case in report["cases"]] != definitions:
        raise ValueError("Reference case content/order mismatch")
    if report.get("loaded_state_integrity", {}).get("verified") is not True:
        raise ValueError("Reference lacks verified loaded checkpoint state")
    runtime = report["runtime"]
    if runtime["device"] != "cpu" or runtime["weight_dtype"] != "torch.float32" or runtime["autocast"]:
        raise ValueError("Reference is not CPU FP32")
    ids = set()
    for case in report["cases"]:
        case_id = case["id"]
        if case_id in ids or Path(case_id).name != case_id or case_id in (".", ".."):
            raise ValueError("Duplicate or unsafe case identity")
        ids.add(case_id)
        checked_file(reference.parent, case["answers"]["file"], case["answers"]["sha256"])
        for call in case["forward_calls"]:
            if set(call) != set(INPUTS + OUTPUTS):
                raise ValueError("Incomplete reference tensor inventory")
            for tensor in call.values():
                checked_file(reference.parent / case_id, tensor["file"], tensor["sha256"])
    return report, manifest

def read_lease(path, digest, now=None):
    if not path or not digest:
        raise Blocked("Externally verified exclusive device lease and digest required")
    if sha256_file(path) != digest:
        raise ValueError("Lease evidence hash mismatch")
    lease = json.loads(Path(path).read_text(encoding="utf-8"))
    if lease.get("schema_version") != 1 or lease.get("verified") is not True or lease.get("exclusive") is not True:
        raise Blocked("Lease must assert verified exclusive allocation")
    for key in ("lease_id", "device_serial", "device_id"):
        if lease.get(key) is None or not str(lease.get(key, "")).strip():
            raise Blocked(f"Missing lease {key}")
    if lease.get("host", "").lower() != socket.gethostname().lower():
        raise Blocked("Lease belongs to another host")
    expiry = datetime.fromisoformat(lease["expires_at"].replace("Z", "+00:00"))
    if expiry.tzinfo is None or expiry <= (now or datetime.now(timezone.utc)):
        raise Blocked("Lease expired or lacks timezone")
    visible = [value.strip() for value in os.environ.get("TT_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if visible != [str(lease["device_id"])]:
        raise Blocked("TT_VISIBLE_DEVICES must name exactly the leased device")
    return lease

def compiler_installation(expected_commit=TT_XLA_AUDITED_COMMIT):
    if expected_commit not in (TT_XLA_AUDITED_COMMIT, OFFLINE_NIGHTLY_COMMIT):
        raise Blocked("Compiler revision has not been audited")
    # Inspect metadata without importing packages that may initialize a runtime.
    modules = ("torch_xla", "torch_plugin_tt", "pjrt_plugin_tt", "tt_torch")
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        raise Blocked("Missing TT compiler modules: " + ", ".join(missing))
    try:
        distribution = importlib.metadata.distribution("pjrt-plugin-tt")
    except importlib.metadata.PackageNotFoundError as exc:
        raise Blocked("pjrt-plugin-tt distribution metadata is unavailable") from exc
    if not any(ep.group == "torch_xla.plugins" and ep.name == "tt" and ep.value == "torch_plugin_tt:TTPlugin"
               for ep in distribution.entry_points):
        raise Blocked("TT PyTorch/XLA plugin registration is absent or differs from audited source")
    summary = distribution.metadata.get("Summary", "")
    commit = re.search(r"(?:^|[, ])commit=([0-9a-f]{40})(?:[, ]|$)", summary)
    if not commit or commit.group(1) != expected_commit:
        raise Blocked("TT compiler build commit is not the audited pinned revision")
    if expected_commit == OFFLINE_NIGHTLY_COMMIT and distribution.version != OFFLINE_NIGHTLY_VERSION:
        raise Blocked("Offline nightly requires its exact audited wheel version")
    return {"version": distribution.version, "commit": commit.group(1)}

def verify_visibility_proof(lease_path, lease, compiler):
    evidence = lease.get("visibility_proof")
    if not isinstance(evidence, dict) or not evidence.get("file") or not evidence.get("sha256"):
        raise Blocked("External pin-version device visibility proof is required before runtime imports")
    path = checked_file(Path(lease_path).parent, evidence["file"], evidence["sha256"])
    proof = json.loads(path.read_text(encoding="utf-8"))
    if proof.get("verified") is not True or proof.get("initialization_isolated") is not True or proof.get("other_devices_untouched") is not True:
        raise Blocked("Visibility proof must verify initialization isolation and untouched other devices")
    for field in ("host", "device_serial"):
        if proof.get(field) != lease.get(field):
            raise Blocked("Visibility proof and lease identity mismatch")
    if proof.get("tt_xla_commit") != compiler["commit"] or proof.get("plugin_version") != compiler["version"]:
        raise Blocked("Visibility proof does not cover the installed compiler build")
    bdf = proof.get("device_bdf", "")
    if not re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", bdf):
        raise Blocked("Visibility proof requires a physical PCI BDF")
    if str(lease["device_id"]) != bdf or os.environ.get("TT_VISIBLE_DEVICES") != bdf:
        raise Blocked("Use the verified PCI BDF as lease device_id and TT_VISIBLE_DEVICES")
    return {"sha256": evidence["sha256"], "device_bdf": bdf}

def preflight(lease_path=None, lease_digest=None):
    reasons = []
    try:
        reject_compile_only_for_hardware()
    except Blocked as exc:
        reasons.append(str(exc))
    if platform.system() != "Linux" or not list(Path("/dev/tenstorrent").glob("*")):
        reasons.append("No Linux Tenstorrent device nodes available")
    compiler = None
    try:
        compiler = compiler_installation()
    except Blocked as exc:
        reasons.append(str(exc))
    try:
        lease = read_lease(lease_path, lease_digest)
    except Blocked as exc:
        reasons.append(str(exc))
        lease = None
    proof = None
    if lease and compiler:
        try:
            proof = verify_visibility_proof(lease_path, lease, compiler)
        except Blocked as exc:
            reasons.append(str(exc))
    return {"compiler": compiler, "visibility_proof": proof, "status": "BLOCKED" if reasons else "READY_FOR_DEVICE_CHECK",
            "reasons": reasons, "lease_id": lease["lease_id"] if lease else None,
            "physical_acceptance": False, "graph_executed": False,
            "note": "Readiness inspection only; does not initialize or prove hardware"}

def reject_compile_only_for_hardware():
    if "TT_COMPILE_ONLY_SYSTEM_DESC" in os.environ:
        raise Blocked("Hardware mode refuses TT_COMPILE_ONLY_SYSTEM_DESC; compile-only buffers are not inference")

def compile_only_preflight(descriptor_path, descriptor_sha256, toolchain_commit=TT_XLA_AUDITED_COMMIT):
    if platform.system() != "Linux":
        raise Blocked("Pinned TT compile-only environment requires Linux")
    if list(Path("/dev/tenstorrent").glob("*")):
        raise Blocked("Compile-only must run without exposed Tenstorrent device nodes")
    if os.environ.get("TT_VISIBLE_DEVICES"):
        raise Blocked("Compile-only requires TT_VISIBLE_DEVICES unset")
    if not descriptor_path or descriptor_sha256 not in (P150_DESCRIPTOR_SHA256, MIGRATED_P150_SHA256):
        raise ValueError("Require the audited P150 descriptor and its pinned SHA256")
    path = Path(descriptor_path).resolve()
    if not path.is_file() or sha256_file(path) != descriptor_sha256:
        raise ValueError("P150 descriptor file hash mismatch")
    metadata = {"file": path.name, "sha256": descriptor_sha256,
                "offline_only": True, "current_card_inventory": False}
    if descriptor_sha256 == MIGRATED_P150_SHA256:
        if toolchain_commit != OFFLINE_NIGHTLY_COMMIT:
            raise Blocked("Migrated descriptor is audited only for the exact offline nightly toolchain")
        provenance_path = checked_file(path.parent, "provenance.json", MIGRATED_P150_PROVENANCE_SHA256)
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if (provenance.get("kind") != "migrated_generic_system_descriptor"
                or provenance.get("offline_only") is not True
                or provenance.get("current_card_inventory") is not False
                or provenance.get("hardware_queried") is not False
                or provenance.get("source_tt_xla_commit") != toolchain_commit
                or provenance.get("descriptor") != {"file": path.name, "sha256": descriptor_sha256}):
            raise ValueError("Migrated descriptor provenance identity mismatch")
        metadata.update(kind="migrated_generic_system_descriptor",
                        migration_provenance_sha256=MIGRATED_P150_PROVENANCE_SHA256,
                        migration=provenance)
    else:
        metadata.update(kind="upstream_generic_system_descriptor", upstream_commit=toolchain_commit)
    inherited = os.environ.get("TT_COMPILE_ONLY_SYSTEM_DESC")
    if inherited is not None and Path(inherited).resolve() != path:
        raise Blocked("Inherited compile-only descriptor differs from explicit pinned descriptor")
    return {"compiler": compiler_installation(toolchain_commit),
            "system_descriptor": metadata,
            "device_nodes_exposed": False}

def select_compile_call(reference, case_id, call_index):
    matches = [case for case in reference["cases"] if case["id"] == case_id]
    if len(matches) != 1 or call_index < 0 or call_index >= len(matches[0]["forward_calls"]):
        raise ValueError("Compile-only requires one existing case-id and forward call index")
    return matches[0], matches[0]["forward_calls"][call_index]

def compile_artifact_inventory(output, prefix):
    result = {}
    for suffix in (".ttnn", "_ttnn.mlir", "_ttir.mlir"):
        path = Path(str(prefix) + suffix)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty compiler artifact: {path.name}")
        result[path.name] = {"file": path.relative_to(output).as_posix(),
                             "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result

def compare_arrays(actual, expected, atol, rtol):
    import numpy as np
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape or not actual.size:
        raise ValueError("Output shape mismatch or empty comparison")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Non-finite output")
    delta = actual - expected
    a, e = actual.ravel(), expected.ravel()
    pcc = float(np.corrcoef(a, e)[0, 1]) if a.std() and e.std() else None
    return {"passed": bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
            "max_abs_error": float(np.max(np.abs(delta))),
            "max_relative_error": float(np.max(np.abs(delta) / np.maximum(np.abs(expected), 1e-12))),
            "rmse": float(np.sqrt(np.mean(delta**2))), "pcc": pcc,
            "atol": atol, "rtol": rtol, "elements": int(actual.size)}

def load_call(base, case_id, call):
    import numpy as np
    import torch
    result = {}
    for name, metadata in call.items():
        path = checked_file(base / case_id, metadata["file"], metadata["sha256"])
        array = np.load(path, allow_pickle=False)
        if list(array.shape) != metadata["shape"] or str(array.dtype) != metadata["dtype"]:
            raise ValueError(f"Tensor metadata mismatch: {name}")
        result[name] = torch.from_numpy(array.copy())
        if str(result[name].dtype) != metadata["torch_dtype"]:
            raise ValueError(f"Torch dtype mismatch: {name}")
    return result

def load_pinned_model(manifest, source, checkpoint, progress):
    import torch
    sys.path.insert(0, str(source))
    import laya.agent as upstream_agent
    if Path(upstream_agent.__file__).resolve() != (source / "laya/agent.py").resolve():
        raise ValueError("Imported Laya source mismatch")
    os.environ.pop("LAYA_CPU_AMP", None)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    agent = upstream_agent.Agent(str(checkpoint), device="cpu", compile=False, fast=False)
    if agent.device.type != "cpu" or agent.amp_enabled or checkpoint_files(checkpoint) != manifest["checkpoint"]["files"]:
        raise ValueError("Reference loading changed pinned artifacts or baseline policy")
    progress["torch_version"] = torch.__version__
    model = agent.model.float().eval()
    from laya_tt.integrity import verify_loaded_state
    progress["loaded_state_integrity"] = verify_loaded_state(model, checkpoint, expected_sha256=manifest["checkpoint"]["files"]["model.safetensors"])
    return model

def run_graphs(args, reference, manifest, source, checkpoint, output, progress):
    import torch
    model = load_pinned_model(manifest, source, checkpoint, progress)
    device = None
    if args.mode == "tt-xla":
        reject_compile_only_for_hardware()
        lease = read_lease(args.lease, args.lease_sha256)
        verify_visibility_proof(args.lease, lease, compiler_installation())
        # Explicit registration from the audited TT-XLA package.
        import tt_torch
        import torch_plugin_tt
        if "tt" not in torch._dynamo.list_backends():
            raise Blocked("TT torch.compile backend was not registered")
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        import torch_xla.debug.metrics as xla_metrics
        xr.set_device_type("TT")
        if xr.device_type() != "TT" or xr.global_runtime_device_count() != 1:
            raise Blocked("TT runtime must expose exactly one TT device")
        # fullgraph rejects Dynamo graph breaks instead of accepting Python fallback.
        model.compile(backend="tt", fullgraph=True)
        device = xm.xla_device()
        model = model.to(device)
    rows = []
    for case in reference["cases"]:
        for index, call in enumerate(case["forward_calls"]):
            tensors = load_call(Path(args.reference).parent, case["id"], call)
            inputs = tuple(tensors[name] for name in INPUTS)
            row = {"case": case["id"], "call": index,
                   "input_shapes": {name: list(tensors[name].shape) for name in INPUTS}}
            if args.mode == "torch-export":
                with torch.no_grad():
                    exported = torch.export.export(model, inputs, strict=True)
                    progress["graph_execution_attempted"] = True
                    actual = exported.module()(*inputs)
                    progress["graph_executed"] = True
                row["operators"] = dict(sorted(Counter(str(node.target) for node in exported.graph.nodes if node.op == "call_function").items()))
                graph_path = output / f"{case['id']}-{index:03d}.graph.txt"
                graph_path.write_text(str(exported.graph), encoding="utf-8")
                row["graph_sha256"] = sha256_file(graph_path)
                row["torch_export_succeeded"] = True
                row["tt_compiled"] = False
                if args.save_export:
                    torch.export.save(exported, output / f"{case['id']}-{index:03d}.pt2")
            else:
                read_lease(args.lease, args.lease_sha256)
                progress["graph_execution_attempted"] = True
                progress["device_execution_attempted"] = True
                xla_metrics.clear_all()
                with torch.no_grad():
                    actual = model(*(tensor.to(device) for tensor in inputs))
                    if any(tensor.device.type != "xla" for tensor in actual):
                        raise RuntimeError("TT graph produced non-XLA output; refusing fallback")
                    actual = tuple(tensor.cpu() for tensor in actual)
                    progress["graph_executed"] = True
                    progress["device_execution"] = True
                fallbacks = {name: xla_metrics.counter_value(name) for name in xla_metrics.counter_names() if name.startswith("aten::") and xla_metrics.counter_value(name)}
                if fallbacks:
                    raise RuntimeError(f"CPU fallback counters detected: {fallbacks}")
                row["cpu_fallback_counters"] = fallbacks
                row["tt_compiled"] = True
            row["comparisons"] = {}
            for name, actual_tensor in zip(OUTPUTS, actual):
                predicted, expected = actual_tensor.detach().float().cpu().numpy(), tensors[name].numpy()
                if name == "logits":
                    mask = tensors["marker_mask"].numpy()
                    # Record valid markers separately; padded -1e4 logits cannot dominate PCC.
                    row["comparisons"]["valid_logits"] = compare_arrays(predicted[mask], expected[mask], args.atol, args.rtol)
                row["comparisons"][name] = compare_arrays(predicted, expected, args.atol, args.rtol)
            rows.append(row)
            (output / f"{case['id']}-{index:03d}.result.json").write_text(json.dumps(row, indent=2, allow_nan=False), encoding="utf-8")
    if not rows:
        raise ValueError("Reference contained no executable tensor graphs")
    return rows

def run_compile_only(args, reference, manifest, source, checkpoint, output, progress):
    # One process and fresh cache per shape; the upstream cache is a singleton.
    compile_only_preflight(args.system_desc, args.system_desc_sha256, args.toolchain_commit)
    case, call = select_compile_call(reference, args.case_id, args.call_index)
    os.environ["TT_COMPILE_ONLY_SYSTEM_DESC"] = str(Path(args.system_desc).resolve())
    import torch
    model = load_pinned_model(manifest, source, checkpoint, progress)
    if getattr(args, "precision", "float32") == "mixed-bf16-fp32":
        from probe_bf16_cpu import apply_candidate_policy
        from laya_tt.integrity import verify_loaded_state
        progress["precision_policy"] = {
            "name": "mixed-bf16-fp32",
            "helper_sha256": sha256_file(ROOT / "scripts/probe_bf16_cpu.py"),
            "inventory": apply_candidate_policy(model),
        }
        progress["candidate_loaded_state_integrity"] = verify_loaded_state(
            model, checkpoint,
            expected_sha256=manifest["checkpoint"]["files"]["model.safetensors"])
    import torch_xla
    import torch_xla.runtime as xr
    import torch_xla.core.xla_model as xm
    import tt_torch
    import torch_plugin_tt
    if "tt" not in torch._dynamo.list_backends():
        raise Blocked("TT torch.compile backend was not registered")
    xr.set_device_type("TT")
    cache = output / "compiler-cache"
    xr.initialize_cache(str(cache))
    device = xm.xla_device()
    if xr.device_type() != "TT" or xr.global_runtime_device_count() != 1:
        raise Blocked("P150 compile-only descriptor must expose exactly one virtual TT device")
    model.compile(backend="tt", fullgraph=True)
    model = model.to(device)
    tensors = load_call(Path(args.reference).parent, case["id"], {name: call[name] for name in INPUTS})
    profile_bucket = getattr(args, "profile_bucket", None)
    if profile_bucket is not None:
        from shape_profiles import profile_rows
        if case["id"] != "single-option" or args.call_index != 0:
            raise ValueError("Profile compilation requires the pinned single-option call")
        # Obtain the padding identity from the hash-validated encoder config.
        tokenizer_config = json.loads((checkpoint / "encoder/config.json").read_text())
        pad_id = tokenizer_config["pad_token_id"]
        tokenizer = json.loads((checkpoint / "tokenizer/tokenizer.json").read_text(encoding="utf-8"))
        tokenizer_options = json.loads((checkpoint / "tokenizer/tokenizer_config.json").read_text(encoding="utf-8"))
        pad_name = tokenizer_options["pad_token"]
        pad_ids = {entry["id"] for entry in tokenizer["added_tokens"] if entry["content"] == pad_name}
        if pad_ids != {pad_id}:
            raise ValueError("Pinned tokenizer and encoder padding identities differ")
        profiled, = profile_rows(tuple(tensors[name] for name in INPUTS), pad_id, sequence_bucket=profile_bucket)
        tensors = dict(zip(INPUTS, profiled))
        progress["profile_layout"] = {"helper_sha256": sha256_file(ROOT / "scripts/shape_profiles.py"),
                                      "pad_token_id": pad_id, "bucket": profile_bucket,
                                      "inputs": {name: {"dtype": str(value.dtype), "shape": list(value.shape),
                                          "sha256": hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()}
                                          for name, value in tensors.items()}}
    progress["compilation_attempted"] = True
    with torch.no_grad():
        # Keep lazy outputs alive for the barrier, but never copy/read/compare them.
        dummy_outputs = model(*(tensors[name].to(device) for name in INPUTS))
        torch_xla.sync(wait=True)
    prefix = output / "compiled" / f"{case['id']}-{args.call_index:03d}"
    tt_torch.parse_compiled_artifacts_from_cache_to_disk(str(cache), str(prefix))
    artifacts = compile_artifact_inventory(output, prefix)
    del dummy_outputs
    return {"case": case["id"], "call": args.call_index,
            "profile_bucket": profile_bucket,
            "input_shapes": {name: list(tensors[name].shape) for name in INPUTS},
            "artifacts": artifacts, "numerical_comparison_performed": False,
            "compilation_status": "COMPILED", "physical_acceptance": False}

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "torch-export", "tt-xla", "tt-compile-only"), required=True)
    parser.add_argument("--reference", default="artifacts/reference/cpu/reference.json")
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--manifest", default="configs/checkpoint-lock.json")
    parser.add_argument("--cases", default="configs/reference-cases.json")
    parser.add_argument("--output", default="artifacts/compiler-spike")
    parser.add_argument("--toolchain-commit", choices=(TT_XLA_AUDITED_COMMIT, OFFLINE_NIGHTLY_COMMIT), default=TT_XLA_AUDITED_COMMIT)
    parser.add_argument("--system-desc")
    parser.add_argument("--system-desc-sha256")
    parser.add_argument("--case-id")
    parser.add_argument("--call-index", type=int, default=0)
    parser.add_argument("--profile-bucket", type=int, choices=(32, 64, 128, 256, 512))
    parser.add_argument("--precision", choices=("float32", "mixed-bf16-fp32"), default="float32")
    parser.add_argument("--lease")
    parser.add_argument("--lease-sha256")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--save-export", action="store_true")
    args = parser.parse_args(argv)
    if args.precision != "float32" and args.mode != "tt-compile-only":
        parser.error("Experimental mixed precision is permitted only in offline compilation")
    if args.profile_bucket is not None and args.mode != "tt-compile-only":
        parser.error("Experimental profiles are permitted only in offline compilation")
    for field in ("reference", "manifest", "cases", "output", "lease", "system_desc"):
        value = getattr(args, field)
        if value:
            setattr(args, field, str((ROOT / value).resolve()))
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "mode": args.mode, "physical_acceptance": False,
              "graph_executed": False, "reference_sha256": args.reference_sha256,
              "dtype": args.precision, "status": "FAILED"}
    code = 1
    original_compile_only = os.environ.get("TT_COMPILE_ONLY_SYSTEM_DESC")
    if args.mode == "tt-compile-only":
        report.pop("graph_executed")
        report["compile_only"] = True
        report["compilation_status"] = "NOT_ATTEMPTED"
    try:
        if not math.isfinite(args.atol) or not math.isfinite(args.rtol) or min(args.atol, args.rtol) < 0:
            raise ValueError("Tolerances must be finite and nonnegative")
        reference, manifest = read_reference(args.reference, args.reference_sha256, args.manifest, args.cases)
        report["manifest_sha256"] = sha256_file(args.manifest)
        if args.mode in ("preflight", "tt-xla"):
            if args.toolchain_commit != TT_XLA_AUDITED_COMMIT:
                raise Blocked("Nightly revision is audited for compile-only, not physical execution")
            check = preflight(args.lease, args.lease_sha256)
            report["preflight"] = check
            if check["status"] == "BLOCKED":
                raise Blocked("; ".join(check["reasons"]))
            if args.mode == "preflight":
                report["status"] = check["status"]
                code = 0
        if args.mode == "tt-compile-only":
            report["compile_only_preflight"] = compile_only_preflight(args.system_desc, args.system_desc_sha256, args.toolchain_commit)
            source, checkpoint = validate_manifest(manifest, ROOT)
            report["compilation"] = run_compile_only(args, reference, manifest, source, checkpoint, output, report)
            report["compilation_status"] = "COMPILED"
            report["status"] = "COMPILED"
            report["note"] = "Compiler artifacts only; dummy outputs were not read and no inference occurred"
            code = 0
        elif args.mode != "preflight":
            source, checkpoint = validate_manifest(manifest, ROOT)
            report["graphs"] = run_graphs(args, reference, manifest, source, checkpoint, output, report)
            report["graph_executed"] = True
            passed = all(metric["passed"] for row in report["graphs"] for metric in row["comparisons"].values())
            report["status"] = "PASSED" if passed else "FAILED"
            report["device_execution"] = args.mode == "tt-xla"
            report["note"] = "Full tensor graph experiment only; platform and semantic acceptance remain separate"
            code = 0 if passed else 1
    except Blocked as exc:
        report.update(status="BLOCKED", reason=str(exc))
        code = 2
    except LoadedStateIntegrityError as exc:
        report.update(status="FAILED", error_type=type(exc).__name__, reason=str(exc),
                      loaded_state_integrity=exc.report)
    except Exception as exc:
        report.update(status="FAILED", error_type=type(exc).__name__, reason=str(exc))
    finally:
        if args.mode == "tt-compile-only":
            if report["status"] != "COMPILED":
                report["compilation_status"] = report["status"]
            if original_compile_only is None:
                os.environ.pop("TT_COMPILE_ONLY_SYSTEM_DESC", None)
            else:
                os.environ["TT_COMPILE_ONLY_SYSTEM_DESC"] = original_compile_only
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(output / "report.json"), "physical_acceptance": False}))
    return code

if __name__ == "__main__":
    raise SystemExit(main())
