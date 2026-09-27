"""Observe loaded rotary buffers against an independent formula; no acceptance gate."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def observe(model):
    import numpy as np
    import torch

    config = model.encoder.config
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    if dim <= 0 or dim % 2:
        raise ValueError("Rotary head dimension must be positive and even")
    kinds = {"full_attention", "sliding_attention"}
    if set(config.rope_parameters) != kinds:
        raise ValueError("Unsupported rotary configuration")
    state_keys = set(model.state_dict())
    buffers = {name: tensor for name, tensor in model.named_buffers() if name not in state_keys}
    expected_names = {f"encoder.rotary_emb.{kind}_{suffix}"
                      for kind in kinds for suffix in ("inv_freq", "original_inv_freq")}
    if set(buffers) != expected_names:
        raise ValueError("Nonpersistent buffer inventory differs from audited ModernBERT layout")
    rows = []
    for kind in sorted(kinds):
        params = config.rope_parameters[kind]
        theta = float(params["rope_theta"])
        if params["rope_type"] != "default" or not np.isfinite(theta) or theta <= 1:
            raise ValueError("Only finite default RoPE theta greater than one is supported")
        expected = (1 / np.power(np.float64(theta), np.arange(0, dim, 2, dtype=np.float64) / dim)).astype(np.float32)
        for suffix in ("inv_freq", "original_inv_freq"):
            name = f"encoder.rotary_emb.{kind}_{suffix}"
            tensor = buffers[name]
            if tensor.device.type != "cpu" or tensor.dtype != torch.float32 or list(tensor.shape) != [dim // 2]:
                raise ValueError(f"Unexpected buffer shape, device or dtype: {name}")
            actual = tensor.detach().contiguous().numpy()
            if not np.isfinite(actual).all() or not (actual > 0).all():
                raise ValueError(f"Nonfinite or nonpositive rotary buffer: {name}")
            rows.append({"buffer": name, "shape": list(actual.shape), "dtype": "float32",
                         "theta": theta, "different_elements": int(np.count_nonzero(actual != expected)),
                         "max_abs_error": float(np.max(np.abs(actual.astype(np.float64) - expected))),
                         "max_ulp": int(np.max(np.abs(actual.view(np.int32).astype(np.int64)
                                                     - expected.view(np.int32).astype(np.int64))))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from laya_tt.reference import validate_manifest, checkpoint_files, sha256_file
    from laya_tt.integrity import verify_loaded_state, LoadedStateIntegrityError

    if args.output.exists():
        raise FileExistsError("Refusing to overwrite earlier observations")
    report = {"schema_version": 1, "physical_acceptance": False, "loaded_model_acceptance": False,
              "acceptance_threshold_established": False,
              "comparison": "NumPy float64 formula rounded to float32 versus loaded FP32 buffers",
              "formula": "1 / theta ** (arange(0, head_dim, 2) / head_dim)"}
    try:
        manifest = json.loads((root / "configs/checkpoint-lock.json").read_text(encoding="utf-8"))
        source, checkpoint = validate_manifest(manifest, root)
        report.update(source_commit=manifest["upstream"]["commit"],
                      checkpoint_revision=manifest["checkpoint"]["revision"],
                      manifest_sha256=sha256_file(root / "configs/checkpoint-lock.json"),
                      script_sha256=sha256_file(__file__))
        sys.path.insert(0, str(source))
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.pop("LAYA_CPU_AMP", None)
        import torch
        import laya.agent as upstream
        if Path(upstream.__file__).resolve() != source / "laya/agent.py":
            raise ValueError("Imported source differs from pin")
        torch.manual_seed(0)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        agent = upstream.Agent(str(checkpoint), device="cpu", compile=False, fast=False)
        if agent.device.type != "cpu" or agent.amp_enabled:
            raise ValueError("Unexpected acceleration")
        model = agent.model.float().eval()
        report["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "transformers")}
        report["observations"] = observe(model)
        report["loaded_state_integrity"] = verify_loaded_state(
            model, checkpoint, expected_sha256=manifest["checkpoint"]["files"]["model.safetensors"])
        if checkpoint_files(checkpoint) != manifest["checkpoint"]["files"]:
            raise ValueError("Checkpoint changed during observation")
        report["diagnostic_completed"] = True
    except Exception as exc:
        report["diagnostic_completed"] = False
        report["error_type"] = type(exc).__name__
        if isinstance(exc, LoadedStateIntegrityError):
            report["loaded_state_integrity"] = exc.report
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
