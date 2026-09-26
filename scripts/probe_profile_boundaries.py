"""Measure CPU layout parity at native admission boundaries; no TT execution."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

from compiler_spike import INPUTS, compare_arrays
from probe_bf16_cpu import decoded_metrics
from shape_profiles import profile_rows


def requests():
    question = {"type": "noul", "instructions": "Does the state mention a word?"}
    return [
        ("64-choice-options", {"model": "laya-english", "state": "Choose 7.", "questions": {
            "choice": {"type": "choice", "instructions": "Select the number.",
                       "criteria": {str(i): "" for i in range(64)}}}}),
        ("64-question-rows", {"model": "laya-english", "state": "The word is present.",
                              "questions": {f"q{i:02d}": copy.deepcopy(question) for i in range(64)}}),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve()
    if not output.is_relative_to(root):
        raise ValueError("Output must remain under repository")
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root / "src"))
    from laya_tt.cpu_backend import load_cpu_backend
    from laya_tt.integrity import verify_loaded_state
    from laya_tt.reference import sha256_file, validate_answers, write_tensor

    report = {"schema_version": 1, "status": "FAILED", "physical_acceptance": False,
              "compiler_attempted": False, "serving_promotion_approved": False,
              "precision": "CPU FP32", "atol": 1e-4, "rtol": 1e-4, "cases": []}
    original_forward = None
    try:
        manifest_path = root / "configs/checkpoint-lock.json"
        manifest = json.loads(manifest_path.read_text())
        backend = load_cpu_backend(manifest_path, root=root)
        agent, torch = backend._agent, backend._torch
        original_forward = agent.model.forward
        checkpoint = root / manifest["checkpoint"]["path"]
        digest = manifest["checkpoint"]["files"]["model.safetensors"]
        report.update(script_sha256=sha256_file(__file__),
                      profile_helper_sha256=sha256_file(root / "scripts/shape_profiles.py"),
                      manifest_sha256=sha256_file(manifest_path), torch_version=torch.__version__,
                      initialization_warnings=backend.initialization_warnings,
                      loaded_state_integrity=verify_loaded_state(agent.model, checkpoint, expected_sha256=digest))
        for name, request in requests():
            prepared = backend.prepare(request)
            payload = prepared.payload
            directory = output / name
            directory.mkdir()
            baseline_calls, profile_calls = [], []

            def baseline_forward(*inputs):
                result = original_forward(*inputs)
                baseline_calls.append(tuple(value.detach().clone() for value in result))
                return result

            def profiled_forward(*inputs):
                results = []
                for profile in profile_rows(inputs, agent.tok.pad_token_id):
                    results.append(original_forward(*profile))
                result = (torch.cat([value[0][:, :inputs[2].shape[1]] for value in results]),
                          torch.cat([value[1] for value in results]))
                profile_calls.append(tuple(value.detach().clone() for value in result))
                return result

            agent.model.forward = baseline_forward
            expected = backend.execute(payload)
            agent.model.forward = profiled_forward
            actual = backend.execute(payload)
            if len(baseline_calls) != 1 or len(profile_calls) != 1:
                raise ValueError("Expected exactly one backend forward per layout")
            validate_answers(actual, request["questions"])
            if actual["encoded_tokens"] != expected["encoded_tokens"] or actual["encoded_tokens"] != prepared.encoded_tokens:
                raise ValueError("Layout changed encoded-token accounting")
            row = {"id": name, "request": request, "encoded_tokens": prepared.encoded_tokens,
                   "input_shapes": {key: list(payload.batch[key].shape) for key in INPUTS},
                   "comparisons": {}, "artifacts": {}, "decoded": decoded_metrics(actual, expected)}
            for key, baseline, profiled in zip(("logits", "act_logits"), baseline_calls[0], profile_calls[0]):
                row["comparisons"][key] = compare_arrays(profiled.numpy(), baseline.numpy(), 1e-4, 1e-4)
                for layout, tensor in (("baseline", baseline), ("profiled", profiled)):
                    row["artifacts"][f"{layout}_{key}"] = write_tensor(directory / f"{layout}-{key}.npy", tensor)
            (directory / "answers.json").write_text(json.dumps({"baseline": expected, "profiled": actual}, indent=2, allow_nan=False) + "\n")
            row["answers_sha256"] = sha256_file(directory / "answers.json")
            report["cases"].append(row)
        report["final_loaded_state_integrity"] = verify_loaded_state(agent.model, checkpoint, expected_sha256=digest)
        report["status"] = "OBSERVED"
    except Exception as exc:
        report.update(error_type=type(exc).__name__, reason=str(exc))
    finally:
        if original_forward is not None:
            agent.model.forward = original_forward
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "physical_acceptance": False}))
    return 0 if report["status"] == "OBSERVED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
