"""CPU-only experiment for finite, single-row compiler shapes; no serving promotion."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from compiler_spike import INPUTS, checked_file, compare_arrays, load_call, read_reference
from probe_bf16_cpu import BASELINE_SHA, decoded_metrics

BUCKETS = (32, 64, 128, 256, 512)
MARKERS = 64


def profile_rows(inputs, pad_token_id):
    """Preserve masked positions and row order; never truncate or change accounting."""
    import torch

    ids, attention, positions, markers, qtype = inputs
    if ids.ndim != 2 or attention.shape != ids.shape:
        raise ValueError("Invalid token shape")
    rows, width = ids.shape
    if not 1 <= rows <= 64 or not 1 <= width <= BUCKETS[-1]:
        raise ValueError("Input outside experimental profile bounds")
    if positions.ndim != 2 or positions.shape != markers.shape or positions.shape[0] != rows:
        raise ValueError("Invalid marker shape")
    if not 1 <= positions.shape[1] <= MARKERS or qtype.shape != (rows,):
        raise ValueError("Input outside experimental marker/type bounds")
    if any(value.device.type != "cpu" for value in inputs):
        raise ValueError("This experiment accepts CPU tensors only")
    if markers.dtype != torch.bool or not torch.all((attention == 0) | (attention == 1)):
        raise ValueError("Invalid masks")
    bucket = next(size for size in BUCKETS if size >= width)
    for row in range(rows):
        padded_ids = ids.new_full((1, bucket), pad_token_id)
        padded_attention = attention.new_zeros((1, bucket))
        padded_positions = positions.new_zeros((1, MARKERS))
        padded_markers = markers.new_zeros((1, MARKERS))
        padded_ids[:, :width] = ids[row:row + 1]
        padded_attention[:, :width] = attention[row:row + 1]
        padded_positions[:, :positions.shape[1]] = positions[row:row + 1]
        padded_markers[:, :markers.shape[1]] = markers[row:row + 1]
        yield (padded_ids, padded_attention, padded_positions, padded_markers, qtype[row:row + 1].clone())


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
    from laya_tt.reference import validate_manifest, checkpoint_files, sha256_file, validate_answers
    from laya_tt.integrity import verify_loaded_state

    report = {"schema_version": 1, "status": "FAILED", "physical_acceptance": False,
              "compiler_attempted": False, "serving_promotion_approved": False,
              "precision": "CPU FP32", "sequence_buckets": BUCKETS, "marker_width": MARKERS,
              "atol": 1e-4, "rtol": 1e-4, "baseline_sha256": BASELINE_SHA, "cases": []}
    try:
        reference_path = root / "tests/fixtures/cpu-reference/reference.json"
        reference, manifest = read_reference(reference_path, BASELINE_SHA, root / "configs/checkpoint-lock.json", root / "configs/reference-cases.json")
        source, checkpoint = validate_manifest(manifest, root)
        sys.path.insert(0, str(source))
        import torch
        import laya.agent as upstream

        if Path(upstream.__file__).resolve() != source / "laya/agent.py":
            raise ValueError("Upstream import differs from pin")
        os.environ.pop("LAYA_CPU_AMP", None)
        torch.manual_seed(0)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        agent = upstream.Agent(str(checkpoint), device="cpu", compile=False, fast=False)
        if agent.device.type != "cpu" or agent.amp_enabled or checkpoint_files(checkpoint) != manifest["checkpoint"]["files"]:
            raise ValueError("Unexpected acceleration or changed checkpoint")
        model = agent.model.float().eval()
        digest = manifest["checkpoint"]["files"]["model.safetensors"]
        report.update(script_sha256=sha256_file(__file__), torch_version=torch.__version__,
                      manifest_sha256=sha256_file(root / "configs/checkpoint-lock.json"),
                      loaded_state_integrity=verify_loaded_state(model, checkpoint, expected_sha256=digest))
        original_forward = model.forward
        agent._infer = lambda batch: model(*(batch[name] for name in INPUTS))
        for case in reference["cases"]:
            row = {"id": case["id"], "forwards": []}

            def forward(*inputs, **kwargs):
                index = len(row["forwards"])
                if kwargs or len(inputs) != len(INPUTS) or index >= len(case["forward_calls"]):
                    raise ValueError("Unexpected forward invocation")
                baseline = load_call(reference_path.parent, case["id"], case["forward_calls"][index])
                if any(value.dtype != baseline[name].dtype or not torch.equal(value, baseline[name]) for name, value in zip(INPUTS, inputs)):
                    raise ValueError("Input differs from pinned reference")
                # Padding is computational overhead, never additional encoded-token usage.
                profiles = list(profile_rows(inputs, agent.tok.pad_token_id))
                results = [original_forward(*profile) for profile in profiles]
                logits = torch.cat([result[0][:, :inputs[2].shape[1]] for result in results])
                actions = torch.cat([result[1] for result in results])
                comparisons = {name: compare_arrays(actual.detach().numpy(), baseline[name].numpy(), 1e-4, 1e-4)
                               for name, actual in (("logits", logits), ("act_logits", actions))}
                row["forwards"].append({"outputs": comparisons, "profile_shapes": [list(p[0].shape) for p in profiles],
                                        "encoded_tokens": int(inputs[1].sum()),
                                        "padded_tokens": sum(p[0].numel() for p in profiles)})
                return logits, actions

            model.forward = forward
            definition = case["input"]
            with torch.no_grad():
                actual = agent.predict_batch(definition["states"], definition["questions"]) if "states" in definition else [agent.predict(definition["state"], definition["questions"])]
            if len(row["forwards"]) != len(case["forward_calls"]):
                raise ValueError("Forward count differs from reference")
            for answers in actual:
                validate_answers(answers, definition["questions"])
            expected = json.loads(checked_file(reference_path.parent, case["answers"]["file"], case["answers"]["sha256"]).read_text())
            row["decoded"] = decoded_metrics(actual, expected)
            report["cases"].append(row)
        report["final_loaded_state_integrity"] = verify_loaded_state(model, checkpoint, expected_sha256=digest)
        report["status"] = "OBSERVED"
    except Exception as exc:
        report.update(error_type=type(exc).__name__, reason=str(exc))
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "physical_acceptance": False}))
    return 0 if report["status"] == "OBSERVED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
