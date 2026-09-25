"""Explicit mixed BF16/FP32 CPU candidate; observation only, no compiler or fallback."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

BASELINE_SHA = "d1cb5aaaebdc21b5b8c6db1287811fb098659370978e9d1f50e593b329acb085"
INPUTS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")


def buffer_inventory(model):
    import torch
    result = {}
    for name, value in model.named_buffers():
        if value.is_floating_point() and value.dtype != torch.float32:
            raise ValueError(f"Expected FP32 buffer: {name}")
        array = value.detach().cpu().contiguous().numpy()
        result[name] = {"dtype": str(value.dtype), "shape": list(value.shape),
                        "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest()}
    return result


def apply_candidate_policy(model):
    import torch
    before = buffer_inventory(model)
    parameters = {}
    for name, parameter in model.named_parameters():
        if parameter.device.type != "cpu" or parameter.dtype != torch.float32:
            raise ValueError(f"Policy requires verified CPU FP32 parameters: {name}")
        if name.startswith("act_head."):
            target = torch.float32
        elif name.startswith(("encoder.", "head.", "type_emb.", "scorer.")):
            target = torch.bfloat16
        else:
            raise ValueError(f"Unknown parameter outside audited policy: {name}")
        parameter.data = parameter.detach().to(dtype=target)
        parameters[name] = {"dtype": str(parameter.dtype), "shape": list(parameter.shape)}
    if buffer_inventory(model) != before:
        raise ValueError("Precision conversion changed a buffer")
    return {"parameters": parameters, "buffers": before}


def tensor_metadata(value):
    import torch
    if isinstance(value, torch.Tensor):
        return [{"dtype": str(value.dtype), "shape": list(value.shape), "device": value.device.type}]
    if isinstance(value, dict):
        return [item for entry in value.values() for item in tensor_metadata(entry)]
    if isinstance(value, (list, tuple)):
        return [item for entry in value for item in tensor_metadata(entry)]
    return []


def install_dtype_hooks(model, events):
    import torch
    modules = {"encoder": (model.encoder, None, torch.bfloat16),
               "type_emb": (model.type_emb, None, torch.bfloat16),
               "scorer": (model.scorer, torch.bfloat16, torch.bfloat16),
               "act_head": (model.act_head, torch.float32, torch.float32)}
    if model.head is not None:
        modules.update({f"head.layers.{index}": (layer, torch.bfloat16, torch.bfloat16)
                        for index, layer in enumerate(model.head.layers)})
    if hasattr(model.encoder, "rotary_emb"):
        modules["encoder.rotary_emb"] = (model.encoder.rotary_emb, torch.bfloat16, torch.bfloat16)
    handles = []
    for name, (module, expected_input, expected_output) in modules.items():
        def hook(_module, args, output, name=name, expected_input=expected_input, expected_output=expected_output):
            inputs, outputs = tensor_metadata(args), tensor_metadata(output)
            events.append({"module": name, "inputs": inputs, "outputs": outputs})
            floating_inputs = [item for item in inputs if item["dtype"] in ("torch.float32", "torch.bfloat16", "torch.float16", "torch.float64")]
            if expected_input is not None and (not floating_inputs or any(item["dtype"] != str(expected_input) for item in floating_inputs)):
                raise ValueError(f"Candidate input dtype mismatch: {name}")
            if not outputs or any(item["dtype"] != str(expected_output) or item["device"] != "cpu" for item in outputs):
                raise ValueError(f"Candidate output dtype mismatch: {name}")
        handles.append(module.register_forward_hook(hook))
    return handles, set(modules)


def decoded_metrics(actual, expected):
    numeric, nonnumeric = [], []
    def visit(a, e, path):
        if type(a) is not type(e):
            nonnumeric.append({"path": path, "equal": False})
        elif isinstance(a, dict):
            nonnumeric.append({"path": path + ".key_order", "equal": list(a) == list(e)})
            for key in a:
                if key not in e:
                    continue
                visit(a[key], e[key], path + "." + key)
        elif isinstance(a, list):
            nonnumeric.append({"path": path + ".length", "equal": len(a) == len(e)})
            for index, (left, right) in enumerate(zip(a, e)):
                visit(left, right, f"{path}[{index}]")
        elif isinstance(a, (float, int)) and not isinstance(a, bool):
            numeric.append({"path": path, "actual": a, "expected": e, "absolute_error": abs(a - e)})
        else:
            nonnumeric.append({"path": path, "equal": a == e})
    visit(actual, expected, "answers")
    return {"numeric_fields": numeric, "nonnumeric_fields": nonnumeric,
            "max_numeric_absolute_error": max((item["absolute_error"] for item in numeric), default=0),
            "nonnumeric_and_order_exact": all(item["equal"] for item in nonnumeric),
            "decoded_answers_exact": actual == expected}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    if not output.is_relative_to(root):
        raise ValueError("Output must remain under repository")
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))
    from compiler_spike import read_reference, load_call, compare_arrays, checked_file
    from laya_tt.reference import validate_manifest, checkpoint_files, write_tensor, validate_answers, sha256_file
    from laya_tt.integrity import verify_loaded_state, LoadedStateIntegrityError
    report = {"schema_version": 1, "status": "FAILED", "physical_acceptance": False,
              "compiler_attempted": False, "bf16_promotion_approved": False,
              "precision_policy": "BF16 parameters except FP32 action head; all buffers preserved FP32",
              "autocast": False, "baseline_sha256": BASELINE_SHA, "cases": []}
    handles = []
    try:
        reference_path = root / "tests/fixtures/cpu-reference/reference.json"
        reference, manifest = read_reference(reference_path, BASELINE_SHA, root / "configs/checkpoint-lock.json", root / "configs/reference-cases.json")
        source, checkpoint = validate_manifest(manifest, root)
        report.update(manifest_sha256=sha256_file(root / "configs/checkpoint-lock.json"),
                      source_commit=manifest["upstream"]["commit"],
                      checkpoint_revision=manifest["checkpoint"]["revision"],
                      script_sha256=sha256_file(__file__))
        sys.path.insert(0, str(source))
        import numpy as np
        import torch
        import laya.agent as upstream
        if Path(upstream.__file__).resolve() != source / "laya/agent.py":
            raise ValueError("Imported upstream source differs from pin")
        os.environ.pop("LAYA_CPU_AMP", None)
        torch.manual_seed(0)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        agent = upstream.Agent(str(checkpoint), device="cpu", compile=False, fast=False)
        if agent.device.type != "cpu" or agent.amp_enabled or checkpoint_files(checkpoint) != manifest["checkpoint"]["files"]:
            raise ValueError("Unexpected acceleration or changed checkpoint")
        model = agent.model.float().eval()
        digest = manifest["checkpoint"]["files"]["model.safetensors"]
        report["fp32_loaded_state_integrity"] = verify_loaded_state(model, checkpoint, expected_sha256=digest)
        report["dtype_inventory"] = apply_candidate_policy(model)
        report["candidate_loaded_state_integrity"] = verify_loaded_state(model, checkpoint, expected_sha256=digest)
        report["versions"] = {"torch": torch.__version__, "numpy": np.__version__}
        report["calibration"] = {"temperature_raw": agent.temperature_raw, "temperature_effective": agent.temperature,
                                 "by_options_raw": agent.temperature_by_options_raw,
                                 "by_options_effective": agent.temperature_by_options}
        events = []
        report["dtype_hook_events"] = events
        handles, expected_hook_names = install_dtype_hooks(model, events)
        original_forward = model.forward
        # Retain native preprocessing/decoding but bypass upstream's AMP/CPU retry.
        agent._infer = lambda batch: model(*(batch[name] for name in INPUTS))
        for case in reference["cases"]:
            case_dir = output / case["id"]
            case_dir.mkdir()
            row = {"id": case["id"], "forwards": []}
            def forward(*inputs, **kwargs):
                index = len(row["forwards"])
                if kwargs or index >= len(case["forward_calls"]):
                    raise ValueError("Unexpected forward invocation")
                baseline = load_call(reference_path.parent, case["id"], case["forward_calls"][index])
                for name, tensor in zip(INPUTS, inputs):
                    if not torch.equal(tensor, baseline[name]) or tensor.dtype != baseline[name].dtype:
                        raise ValueError(f"Input differs from accepted fixture: {name}")
                first_event = len(events)
                result = original_forward(*inputs)
                observed_hooks = {event["module"] for event in events[first_event:]}
                if observed_hooks != expected_hook_names:
                    raise ValueError("Missing required activation dtype hooks")
                comparison = {"dtype_hooks": events[first_event:], "outputs": {}, "artifacts": {}}
                for name, actual in zip(("logits", "act_logits"), result):
                    if actual.dtype != torch.float32:
                        raise ValueError("Candidate public logits must remain FP32")
                    mask = baseline["marker_mask"] if name == "logits" else torch.ones_like(actual, dtype=torch.bool)
                    comparison["outputs"][name] = compare_arrays(actual[mask].detach().numpy(), baseline[name][mask].numpy(), 1e-4, 1e-4)
                    comparison["artifacts"][name] = write_tensor(case_dir / f"{index:03d}-{name}.npy", actual)
                comparison["action_probability"] = compare_arrays(torch.softmax(result[1], -1).detach().numpy(), torch.softmax(baseline["act_logits"], -1).numpy(), 1e-4, 1e-4)
                row["forwards"].append(comparison)
                return result
            model.forward = forward
            definition = case["input"]
            with torch.no_grad():
                actual = agent.predict_batch(definition["states"], definition["questions"]) if "states" in definition else [agent.predict(definition["state"], definition["questions"])]
            if len(row["forwards"]) != len(case["forward_calls"]):
                raise ValueError("Forward count differs from accepted fixture")
            for answer in actual:
                validate_answers(answer, definition["questions"])
            expected_path = checked_file(reference_path.parent, case["answers"]["file"], case["answers"]["sha256"])
            row["decoded_comparison"] = decoded_metrics(actual, json.loads(expected_path.read_text()))
            answer_path = case_dir / "answers.json"
            answer_path.write_text(json.dumps(actual, indent=2, allow_nan=False) + "\n")
            row["answers_sha256"] = sha256_file(answer_path)
            report["cases"].append(row)
        if buffer_inventory(model) != report["dtype_inventory"]["buffers"]:
            raise ValueError("Forward execution changed preserved buffers")
        report["final_loaded_state_integrity"] = verify_loaded_state(model, checkpoint, expected_sha256=digest)
        report.update(status="OBSERVED", note="Existing FP32 tolerance is diagnostic only; no BF16 acceptance threshold or promotion is implied")
    except LoadedStateIntegrityError as exc:
        report.update(error_type=type(exc).__name__, reason=str(exc), failure_integrity=exc.report)
    except Exception as exc:
        report.update(error_type=type(exc).__name__, reason=str(exc))
    finally:
        for handle in handles:
            handle.remove()
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "physical_acceptance": False}))
    return 0 if report["status"] == "OBSERVED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
