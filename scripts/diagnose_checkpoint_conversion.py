"""CPU-only, per-tensor conversion/copy diagnosis. No model loading or repair."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import struct
import sys

CHECKPOINT_SHA256 = "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"
EXPECTED_TENSORS = 206
EXPECTED_ELEMENTS = 421293830


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def array_sha(array):
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def compare_bits(expected, actual, source, raw_stream=None, data_offset=0):
    import numpy as np
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise ValueError("Shape/dtype mismatch in bit comparison")
    uint = {2: np.uint16, 4: np.uint32}[expected.dtype.itemsize]
    eb, ab = expected.view(uint), actual.view(uint)
    mask = eb != ab
    count = int(np.count_nonzero(mask))
    result = {"matched": count == 0, "mismatched_elements": count,
              "expected_sha256": array_sha(expected), "actual_sha256": array_sha(actual)}
    if count:
        flat = int(np.argmax(mask.reshape(-1)))
        index = tuple(int(i) for i in np.unravel_index(flat, expected.shape))
        source_uint = {2: np.uint16, 4: np.uint32}[source.dtype.itemsize]
        result["first_mismatch"] = {"flat_index": flat, "index": list(index),
            "source_bits": f"0x{int(source.view(source_uint)[index]):0{source.dtype.itemsize * 2}x}",
            "expected_bits": f"0x{int(eb[index]):0{expected.dtype.itemsize * 2}x}",
            "actual_bits": f"0x{int(ab[index]):0{actual.dtype.itemsize * 2}x}"}
        offset = data_offset + flat * source.dtype.itemsize
        if raw_stream is not None:
            raw_stream.seek(offset)
            original_bytes = raw_stream.read(source.dtype.itemsize)
        else:
            original_bytes = source.reshape(-1)[flat:flat + 1].tobytes()
        value = struct.unpack("<e" if source.dtype.itemsize == 2 else "<f", original_bytes)[0]
        scalar_fp32 = struct.pack("<f", value)
        scalar_storage = struct.pack("<e" if actual.dtype.itemsize == 2 else "<f", value)
        scalar_storage_bits = int.from_bytes(scalar_storage, "little")
        result["first_mismatch"]["scalar_oracle"] = {
            "checkpoint_byte_offset": offset, "reread_from_checkpoint": raw_stream is not None,
            "source_storage_hex": original_bytes.hex(), "source_value": value,
            "independent_fp32_bits": f"0x{int.from_bytes(scalar_fp32, 'little'):08x}",
            "numpy_expected_matches_scalar": int(eb[index]) == scalar_storage_bits,
            "observed_matches_scalar": int(ab[index]) == scalar_storage_bits}
    return result


def read_header(stream, file_size):
    header_size = struct.unpack("<Q", stream.read(8))[0]
    if not 2 <= header_size <= min(file_size - 8, 16 * 1024 * 1024):
        raise ValueError("Invalid safetensors header length")
    header_bytes = stream.read(header_size)
    header = json.loads(header_bytes)
    data_start = 8 + header_size
    tensors = {name: item for name, item in header.items() if name != "__metadata__"}
    intervals = []
    for name, item in tensors.items():
        if item["dtype"] not in ("F16", "F32"):
            raise ValueError("Diagnostic only supports pinned F16/F32 checkpoint")
        shape = item["shape"]
        if any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("Invalid tensor shape")
        start, end = item["data_offsets"]
        itemsize = 2 if item["dtype"] == "F16" else 4
        if start < 0 or end < start or data_start + end > file_size or end - start != math.prod(shape) * itemsize:
            raise ValueError("Invalid tensor byte range")
        intervals.append((start, end))
    intervals.sort()
    if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
        raise ValueError("Overlapping tensor data")
    return tensors, data_start, hashlib.sha256(header_bytes).hexdigest()


def diagnose(checkpoint, output):
    import numpy as np
    import safetensors
    from safetensors import safe_open
    import torch
    if output.exists():
        raise ValueError("Diagnostic output must not exist")
    output.mkdir(parents=True)
    report = {"schema_version": 1, "kind": "cpu_checkpoint_conversion_diagnostic",
              "status": "INCOMPLETE", "physical_acceptance": False, "model_repaired": False,
              "tt_runtime_imported": False, "tensors": [],
              "versions": {"python": platform.python_version(), "torch": torch.__version__,
                           "numpy": np.__version__, "safetensors": safetensors.__version__,
                           "os": platform.system()},
              "threads": 1, "deterministic_algorithms": True}
    path = output / "report.json"
    def save():
        path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    try:
        if sys.byteorder != "little":
            raise ValueError("Diagnostic requires the audited little-endian CPU host")
        report["checkpoint_sha256_before"] = file_sha(checkpoint)
        if report["checkpoint_sha256_before"] != CHECKPOINT_SHA256:
            raise ValueError("Pinned checkpoint SHA256 mismatch")
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with checkpoint.open("rb") as raw, safe_open(str(checkpoint), framework="numpy") as numpy_archive, safe_open(str(checkpoint), framework="pt", device="cpu") as torch_archive:
            tensors, data_start, header_sha = read_header(raw, checkpoint.stat().st_size)
            report["header_sha256"] = header_sha
            if set(tensors) != set(numpy_archive.keys()) or set(tensors) != set(torch_archive.keys()):
                raise ValueError("Raw/safetensors key sets differ")
            if len(tensors) != EXPECTED_TENSORS or sum(math.prod(item["shape"]) for item in tensors.values()) != EXPECTED_ELEMENTS:
                raise ValueError("Pinned tensor count/element count mismatch")
            for name in sorted(tensors):
                item = tensors[name]
                start, end = item["data_offsets"]
                raw.seek(data_start + start)
                payload = raw.read(end - start)
                if len(payload) != end - start:
                    raise ValueError("Truncated checkpoint tensor")
                source = np.frombuffer(payload, dtype="<f2" if item["dtype"] == "F16" else "<f4").reshape(item["shape"])
                expected = source.astype(np.float32, copy=True)
                if not np.isfinite(expected).all():
                    raise ValueError("Non-finite checkpoint values")
                row = {"key": name, "source_dtype": item["dtype"], "shape": item["shape"],
                       "elements": int(source.size), "source_raw_sha256": array_sha(source),
                       "source_file_byte_offset": data_start + start, "checks": {}}
                def check(a, b):
                    return compare_bits(a, b, source, raw, data_start + start)
                from_numpy_archive = numpy_archive.get_tensor(name)
                row["checks"]["safetensors_numpy_source"] = check(source, from_numpy_archive)
                del from_numpy_archive
                torch_source = torch_archive.get_tensor(name)
                row["checks"]["safetensors_torch_source"] = check(source, torch_source.numpy())
                converted = torch_source.to(dtype=torch.float32, copy=True)
                row["checks"]["torch_to_float32"] = check(expected, converted.numpy())
                del converted
                destination = torch.empty(item["shape"], dtype=torch.float32, device="cpu")
                destination.copy_(torch_source)
                row["checks"]["torch_copy_into_float32"] = check(expected, destination.numpy())
                del destination
                row["checks"]["torch_source_after_operations"] = check(source, torch_source.numpy())
                del torch_source, expected, source, payload
                report["tensors"].append(row)
                save()
        report["checkpoint_sha256_after"] = file_sha(checkpoint)
        if report["checkpoint_sha256_after"] != CHECKPOINT_SHA256:
            raise ValueError("Checkpoint changed during diagnostic")
        report["checked_tensors"] = len(report["tensors"])
        report["checked_elements"] = sum(row["elements"] for row in report["tensors"])
        failures = [{"key": row["key"], "operation": operation, **result}
                    for row in report["tensors"] for operation, result in row["checks"].items() if not result["matched"]]
        report["mismatches"] = failures
        report["status"] = "MISMATCH" if failures else "MATCHED"
    except Exception as exc:
        report.update(status="FAILED", error_type=type(exc).__name__, reason=str(exc))
    save()
    print(json.dumps({"status": report["status"], "report": str(path), "report_sha256": file_sha(path)}))
    return 0 if report["status"] == "MATCHED" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    return diagnose(Path(args.checkpoint).resolve(), Path(args.output).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
