"""Pinned, CPU-only reference capture. Heavy dependencies are imported on demand."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import warnings

UPSTREAM_COMMIT = "970dc8c5f63d7b886a68409493f37d569424f933"

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def resolve_path(root, value):
    path = Path(value)
    return (Path(root) / path).resolve() if not path.is_absolute() else path.resolve()

def checkpoint_files(path):
    return {p.relative_to(path).as_posix(): sha256_file(p)
            for p in sorted(Path(path).rglob("*"))
            if p.is_file() and ".cache" not in p.relative_to(path).parts
            and ".huggingface" not in p.relative_to(path).parts
            and ".git" not in p.relative_to(path).parts}

def validate_manifest(manifest, root):
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint lock schema")
    upstream = manifest["upstream"]
    if upstream["commit"] != UPSTREAM_COMMIT:
        raise ValueError("Upstream revision differs from audited source")
    source = resolve_path(root, upstream["path"])
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if head != upstream["commit"]:
        raise ValueError("Upstream checkout HEAD mismatch")
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty.strip():
        raise ValueError("Upstream tracked source is modified")
    checkpoint = manifest["checkpoint"]
    if not re.fullmatch(r"[0-9a-f]{40}", checkpoint["revision"]):
        raise ValueError("Checkpoint revision must be an immutable full commit SHA")
    if not checkpoint.get("repo_id"):
        raise ValueError("Checkpoint repository identity is required")
    path = resolve_path(root, checkpoint["path"])
    expected = checkpoint["files"]
    required = {"rl_agent_config.json", "model.safetensors", "encoder/config.json", "tokenizer/tokenizer.json"}
    if not required <= set(expected):
        raise ValueError("Checkpoint lock lacks required artifacts")
    if not all(re.fullmatch(r"[0-9a-f]{64}", h) for h in expected.values()):
        raise ValueError("Invalid artifact SHA256")
    if checkpoint_files(path) != expected:
        raise ValueError("Checkpoint files differ from lock (missing, added, or changed)")
    if manifest.get("runtime") != {"device": "cpu", "dtype": "float32"}:
        raise ValueError("Reference baseline must explicitly select CPU float32")
    return source, path

def validate_answers(result, questions):
    if set(result["answers"]) != set(questions):
        raise ValueError("Question/result identity mismatch")
    for key, question in questions.items():
        answer = result["answers"][key]
        kind = question["type"]
        if answer["type"] != kind:
            raise ValueError("Answer type mismatch")
        for field in ("answer_confidence",):
            if not 0 <= answer[field] <= 1:
                raise ValueError("Invalid confidence")
        if not 0 <= answer["action"]["act_probability"] <= 1:
            raise ValueError("Invalid action probability")
        if kind in ("choice", "score"):
            probs = answer["probabilities"]
            if not probs or any(not 0 <= p <= 1 for p in probs.values()) or abs(sum(probs.values()) - 1) > len(probs) * 0.000051:
                raise ValueError("Invalid probability distribution")
            if kind == "choice" and answer["choice"] not in probs:
                raise ValueError("Choice outside distribution")
            if kind == "score" and not 0 <= answer["score"] <= len(probs) - 1:
                raise ValueError("Score outside rubric")
        elif kind == "noul":
            if not 0 <= answer["noul"] <= 1:
                raise ValueError("Invalid noul probability")
        else:
            raise ValueError("Unsupported answer kind")

def write_tensor(path, tensor):
    import numpy as np
    array = tensor.detach().cpu().contiguous().numpy()
    array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    with Path(path).open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    return {"file": Path(path).name, "sha256": sha256_file(path),
            "shape": list(array.shape), "dtype": str(array.dtype), "torch_dtype": str(tensor.dtype)}

def capture_reference(manifest_path, cases_path, output, root):
    manifest_path, cases_path, output = map(Path, (manifest_path, cases_path, output))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source, checkpoint = validate_manifest(manifest, root)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty; refusing to replace evidence")
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(source))
    import torch
    import laya.agent as upstream_agent
    if Path(upstream_agent.__file__).resolve() != (source / "laya/agent.py").resolve():
        raise ValueError("Imported Laya does not match pinned source")
    # Upstream's compatibility helper mutates cached tokenizer files. Refuse a mutation
    # by checking artifacts again after initialization before producing any outputs.
    os.environ.pop("LAYA_CPU_AMP", None)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    with warnings.catch_warnings(record=True) as initialization_warnings:
        warnings.simplefilter("always")
        agent = upstream_agent.Agent(str(checkpoint), device="cpu", compile=False, fast=False)
    if agent.device.type != "cpu" or agent.amp_enabled:
        raise ValueError("Reference unexpectedly selected acceleration")
    if checkpoint_files(checkpoint) != manifest["checkpoint"]["files"]:
        raise ValueError("Upstream initialization mutated checkpoint; normalize and repin deliberately")
    agent.model.float().eval()
    from .integrity import verify_loaded_state
    integrity = verify_loaded_state(agent.model, checkpoint, expected_sha256=manifest["checkpoint"]["files"]["model.safetensors"])
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "manifest": manifest, "loaded_state_integrity": integrity,
              "manifest_sha256": sha256_file(manifest_path), "cases_sha256": sha256_file(cases_path),
              "runtime": {"device": "cpu", "weight_dtype": "torch.float32", "autocast": False,
                          "deterministic_algorithms": True, "threads": 1,
                          "versions": {name: importlib.metadata.version(name) for name in
                                       ("torch", "transformers", "numpy", "safetensors")}},
              "calibration": {
                  "temperature_raw": agent.temperature_raw,
                  "temperature_effective": agent.temperature,
                  "by_options_raw": agent.temperature_by_options_raw,
                  "by_options_effective": agent.temperature_by_options,
                  "initialization_warnings": [str(w.message) for w in initialization_warnings],
              },
              "physical_acceptance": False, "cases": []}
    original = agent.model.forward
    try:
        for case in cases["cases"]:
            case_dir = output / case["id"]
            case_dir.mkdir()
            calls = []
            def forward(*args, **kwargs):
                result = original(*args, **kwargs)
                call_id = len(calls)
                names = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
                tensors = dict(zip(names, args))
                tensors.update({k: v for k, v in kwargs.items() if k in names})
                tensors.update(zip(("logits", "act_logits"), result))
                calls.append({name: write_tensor(case_dir / f"{call_id:03d}-{name}.npy", tensor)
                              for name, tensor in tensors.items()})
                return result
            agent.model.forward = forward
            if "states" in case:
                results = agent.predict_batch(case["states"], case["questions"])
            else:
                results = [agent.predict(case["state"], case["questions"])]
            for result in results:
                validate_answers(result, case["questions"])
            payload = case_dir / "answers.json"
            payload.write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            report["cases"].append({"id": case["id"], "input": case, "forward_calls": calls,
                                    "answers": {"file": f"{case['id']}/answers.json", "sha256": sha256_file(payload)}})
    finally:
        agent.model.forward = original
    (output / "reference.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
