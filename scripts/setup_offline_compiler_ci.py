"""Install the audited Linux compiler lock without an unpinned extra index."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

VERSION = "1.5.0.dev20260831000501"
COMMIT = "873c53c4ff84bd91112a1f43e788cfed75aaa914"
WHEEL_URL = "https://pypi.eng.aws.tenstorrent.com/pjrt-plugin-tt/pjrt_plugin_tt-1.5.0.dev20260831000501-cp312-cp312-manylinux_2_34_x86_64.whl"
WHEEL_SHA256 = "bfa7dd0834abb55cf37bf2fe57214d0a1074738155175cd0e177b6e1d5408bc1"


def resolved_lock(text):
    original = f"pjrt-plugin-tt=={VERSION}"
    lines = text.splitlines()
    if lines.count(original) != 1:
        raise ValueError("Expected exactly one audited nightly entry in compiler lock")
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        if "==" not in line and not (" @ https://" in line and "#sha256=" in line):
            raise ValueError("Compiler lock contains an unpinned or non-HTTPS requirement")
    return "\n".join(f"pjrt-plugin-tt @ {WHEEL_URL}#sha256={WHEEL_SHA256}" if line == original else line for line in lines) + "\n"


def assert_device_free():
    import os
    if platform.system() != "Linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Requires Linux x86_64 / Python 3.12")
    if list(Path("/dev/tenstorrent").glob("*")):
        raise RuntimeError("TT device nodes must not be exposed")
    if os.environ.get("TT_VISIBLE_DEVICES") or "TT_COMPILE_ONLY_SYSTEM_DESC" in os.environ:
        raise RuntimeError("Inherited accelerator selectors must be absent")


def run(command, **kwargs):
    return subprocess.run(command, check=True, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", default="requirements-compiler-linux-py312.lock")
    parser.add_argument("--output", default="artifacts/offline-ci/environment")
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "FAILED", "physical_acceptance": False,
              "compiler_commit": COMMIT, "wheel_url": WHEEL_URL, "wheel_sha256": WHEEL_SHA256}
    try:
        assert_device_free()
        cpuinfo = Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0]
        report["host_cpu"] = {key.strip(): value.strip() for line in cpuinfo.splitlines()
                              if ":" in line for key, value in [line.split(":", 1)]
                              if key.strip() in ("vendor_id", "model name", "flags")}
        report["python"] = platform.python_version()
        report["kernel_release"] = platform.release()
        lock = Path(args.lock)
        generated = resolved_lock(lock.read_text(encoding="utf-8"))
        resolved = output / "resolved-compiler.lock"
        resolved.write_text(generated, encoding="utf-8")
        report["source_lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()
        report["resolved_lock_sha256"] = hashlib.sha256(resolved.read_bytes()).hexdigest()
        # Every dependency is listed by the known-good environment. No resolver
        # upgrade or extra-index substitution is allowed; pip check is mandatory.
        run([sys.executable, "-m", "pip", "install", "--no-deps", "--only-binary=:all:",
             "--index-url", "https://pypi.org/simple", "--report", str(output / "pip-install.json"),
             "-r", str(resolved)])
        run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "."])
        run([sys.executable, "-m", "pip", "check"])
        frozen = run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
        (output / "environment.lock").write_text(frozen.stdout, encoding="utf-8")
        assert_device_free()
        # Same import/registration probe that passed locally; no device API calls.
        import torch
        import torch_xla
        import torch_plugin_tt
        import pjrt_plugin_tt
        import tt_torch
        installed = importlib.metadata.version("pjrt-plugin-tt")
        summary = importlib.metadata.metadata("pjrt-plugin-tt")["Summary"]
        if installed != VERSION or f"commit={COMMIT}," not in summary:
            raise RuntimeError("Installed plugin provenance differs from audited nightly")
        if torch.__version__ != "2.11.0+cpu" or "tt" not in torch._dynamo.list_backends():
            raise RuntimeError("Torch version or TT backend registration mismatch")
        report.update(status="READY_FOR_OFFLINE_COMPILATION", plugin_version=installed,
                      plugin_summary=summary, torch_version=torch.__version__,
                      device_nodes_exposed=False, backend_registered=True)
    except Exception as exc:
        report.update(error_type=type(exc).__name__, reason=str(exc))
        raise
    finally:
        (output / "setup.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
