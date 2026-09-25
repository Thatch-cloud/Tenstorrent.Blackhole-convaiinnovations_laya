"""Exact loaded checkpoint verification; no repair or tolerance policy."""
from __future__ import annotations
import hashlib
from pathlib import Path

class LoadedStateIntegrityError(RuntimeError):
    def __init__(self, report):
        self.report = report
        keys = [entry["key"] for entry in report.get("mismatches", [])[:8]]
        super().__init__("Loaded checkpoint state failed exact integrity verification"
                         + (": " + ", ".join(keys) if keys else ""))

def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _numpy_values(tensor):
    """Read raw BF16 bits without relying on Torch's numeric conversion kernel."""
    import numpy as np
    import torch
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        bits = tensor.view(torch.uint16).numpy().astype(np.uint32)
        return (bits << 16).view(np.float32)
    return tensor.numpy()

def _promote_independently(source, target_dtype):
    import numpy as np
    import torch
    values = _numpy_values(source)
    if target_dtype == torch.bfloat16:
        float_values = values.astype(np.float32)
        if not np.isfinite(float_values).all():
            raise ValueError("Non-finite checkpoint weights cannot be promoted as a valid baseline")
        bits = float_values.view(np.uint32)
        rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
        return (rounded >> 16).astype(np.uint16)
    dtypes = {torch.float64: np.float64, torch.float32: np.float32, torch.float16: np.float16,
              torch.int64: np.int64, torch.int32: np.int32, torch.int16: np.int16,
              torch.int8: np.int8, torch.uint8: np.uint8, torch.bool: np.bool_}
    if target_dtype not in dtypes:
        raise ValueError(f"Unsupported verification dtype: {target_dtype}")
    promoted = values.astype(dtypes[target_dtype])
    if np.issubdtype(promoted.dtype, np.floating) and not np.isfinite(promoted).all():
        raise ValueError("Non-finite checkpoint weights cannot be used as a valid baseline")
    return promoted

def verify_loaded_state(model, checkpoint, *, expected_sha256=None):
    """Check every state_dict entry exactly against independent checkpoint promotion.

    Raises LoadedStateIntegrityError on mismatch. The caller must pin source/model
    identity separately. expected_sha256 should come from the verified artifact lock.
    Derived nonpersistent buffers are reported as unchecked, not silently certified.
    """
    import numpy as np
    import torch
    from safetensors import safe_open
    path = Path(checkpoint)
    if path.is_dir():
        path = path / "model.safetensors"
    digest = _sha256(path)
    report = {"verified": False, "checkpoint_sha256": digest,
              "comparison": "exact_numpy_promotion", "checked_tensors": 0,
              "checked_elements": 0, "mismatch_tensors": 0, "mismatches": [],
              "unchecked_nonpersistent_buffers": []}
    if expected_sha256 is not None and digest != expected_sha256:
        report["mismatches"].append({"key": "<artifact>", "reason": "checkpoint_sha256_mismatch"})
        report["mismatch_tensors"] = 1
        raise LoadedStateIntegrityError(report)
    state = model.state_dict()
    report["unchecked_nonpersistent_buffers"] = sorted(name for name, _ in model.named_buffers() if name not in state)
    with safe_open(str(path), framework="pt", device="cpu") as archive:
        source_keys = set(archive.keys())
        model_keys = set(state)
        for key in sorted(source_keys - model_keys):
            report["mismatches"].append({"key": key, "reason": "checkpoint_key_missing_from_model"})
        for key in sorted(model_keys - source_keys):
            report["mismatches"].append({"key": key, "reason": "model_key_missing_from_checkpoint"})
        for key in sorted(source_keys & model_keys):
            actual = state[key].detach()
            source = archive.get_tensor(key)
            metadata = {"key": key, "model_dtype": str(actual.dtype), "checkpoint_dtype": str(source.dtype),
                        "model_shape": list(actual.shape), "checkpoint_shape": list(source.shape)}
            if actual.shape != source.shape:
                report["mismatches"].append(dict(metadata, reason="shape_mismatch"))
                continue
            if actual.device.type != "cpu":
                report["mismatches"].append(dict(metadata, reason="verify_on_cpu_before_device_transfer"))
                continue
            try:
                expected = _promote_independently(source, actual.dtype)
                observed = actual.contiguous().view(torch.uint16).numpy() if actual.dtype == torch.bfloat16 else _numpy_values(actual)
            except (TypeError, ValueError, RuntimeError) as exc:
                report["mismatches"].append(dict(metadata, reason="dtype_or_data_invalid", error=str(exc)))
                continue
            different = observed != expected
            count = int(np.count_nonzero(different))
            report["checked_tensors"] += 1
            report["checked_elements"] += actual.numel()
            if count:
                first = list(np.unravel_index(int(np.flatnonzero(different)[0]), different.shape))
                first = [int(index) for index in first]
                if actual.dtype == torch.bfloat16:
                    a = (observed.astype(np.uint32) << 16).view(np.float32)
                    b = (expected.astype(np.uint32) << 16).view(np.float32)
                else:
                    a, b = observed, expected
                finite = np.isfinite(a).all() and np.isfinite(b).all()
                maximum = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if finite else None
                report["mismatches"].append(dict(metadata, reason="value_mismatch", differing_elements=count,
                                                 first_index=first, max_abs_error=maximum))
            del source
    if _sha256(path) != digest:
        report["mismatches"].append({"key": "<artifact>", "reason": "checkpoint_changed_during_verification"})
    report["mismatch_tensors"] = len(report["mismatches"])
    if report["mismatches"]:
        raise LoadedStateIntegrityError(report)
    report["verified"] = True
    return report
