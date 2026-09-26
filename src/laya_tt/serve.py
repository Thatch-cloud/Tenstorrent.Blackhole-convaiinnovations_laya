"""Explicit local CPU-reference process; no accelerator selection or fallback."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

from .bootstrap import RuntimeApplication
from .cpu_backend import load_cpu_backend
from .grants import DecisionGrantVerifier
from .hosting import HostedRuntime
from .service import RuntimeIdentity

REFERENCE_SHA256 = "d1cb5aaaebdc21b5b8c6db1287811fb098659370978e9d1f50e593b329acb085"


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate assignment field")
        result[key] = value
    return result


def _read_json(path, limit=65536, expected_sha256=None):
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("assignment exceeds size limit")
    if expected_sha256 is not None and (not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
            or hashlib.sha256(raw).hexdigest() != expected_sha256):
        raise ValueError("assignment differs from pinned digest")
    return json.loads(raw, object_pairs_hook=_pairs)


def build_cpu_runtime(assignment_path, *, root, assignment_sha256=None):
    """Load an operator-owned assignment and assets, then require golden startup.

    Assignment and root must be protected by the launcher. Public keys authenticate
    grants, not the assignment file itself. This function never downloads assets,
    allocates a device, reconciles an uncertain journal, or issues tenant grants.
    """
    value = _read_json(assignment_path, expected_sha256=assignment_sha256)
    required = {"schema_version", "runtime", "issuer", "host_id", "policy_revision",
                "public_keys", "journal_path", "ledger_socket", "ledger_uid"}
    if not isinstance(value, dict) or set(value) != required or type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("invalid runtime assignment fields")
    runtime = RuntimeIdentity(**value["runtime"])
    if runtime.backend != "cpu-reference":
        raise ValueError("this entry point requires explicit cpu-reference backend")
    for name in ("journal_path", "ledger_socket"):
        if not isinstance(value[name], str) or not Path(value[name]).is_absolute():
            raise ValueError("absolute local runtime paths required")
    if type(value["ledger_uid"]) is not int or not 0 <= value["ledger_uid"] <= 2**32 - 1:
        raise ValueError("invalid ledger peer UID")
    if not isinstance(value["public_keys"], dict) or any(
            not isinstance(raw, str) or not re.fullmatch(r"[0-9a-f]{64}", raw)
            for raw in value["public_keys"].values()):
        raise ValueError("public keys must be 32-byte lowercase hex")
    verifier = DecisionGrantVerifier(
        public_keys={kid: bytes.fromhex(raw) for kid, raw in value["public_keys"].items()},
        issuer=value["issuer"], host_id=value["host_id"], runtime=asdict(runtime),
        policy_revision=value["policy_revision"])
    root = Path(root).resolve()
    reference_path = root / "tests/fixtures/cpu-reference/reference.json"
    reference_bytes = reference_path.read_bytes()
    if hashlib.sha256(reference_bytes).hexdigest() != REFERENCE_SHA256:
        raise ValueError("startup reference differs from pinned reference")
    reference = json.loads(reference_bytes)
    manifest_path = root / "configs/checkpoint-lock.json"
    manifest = _read_json(manifest_path)
    if manifest != reference["manifest"] or runtime.checkpoint_revision != manifest["checkpoint"]["revision"]:
        raise ValueError("assignment or checkpoint lock differs from startup reference")
    case = next(c for c in reference["cases"] if c["id"] == "mixed-question-widths")
    answers_bytes = (reference_path.parent / case["answers"]["file"]).read_bytes()
    if hashlib.sha256(answers_bytes).hexdigest() != case["answers"]["sha256"]:
        raise ValueError("startup answers differ from pinned reference")
    expected = json.loads(answers_bytes)[0]["answers"]
    backend = load_cpu_backend(manifest_path, root=root)
    request = {"model": runtime.model, "state": case["input"]["state"],
               "questions": case["input"]["questions"]}
    prepared = backend.prepare(request)
    application = RuntimeApplication(backend=backend, runtime=runtime, verify_grant=verifier,
        policy_revision=value["policy_revision"], journal_path=Path(value["journal_path"]),
        ledger_socket=Path(value["ledger_socket"]), ledger_uid=value["ledger_uid"])

    def self_test():
        result = backend.execute(prepared.payload)
        return result["answers"] == expected and result["encoded_tokens"] == prepared.encoded_tokens

    return HostedRuntime(application, self_test=self_test)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--assignment-sha256", help="expected exact assignment bytes, supplied by the trusted launcher")
    parser.add_argument("--root", type=Path, required=True, help="pinned source, checkpoint and reference root")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--listen-host", choices=("127.0.0.1", "0.0.0.0"), default="127.0.0.1",
                        help="pod interface requires explicit host-owned network policy")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    import uvicorn
    app = build_cpu_runtime(args.assignment, root=args.root, assignment_sha256=args.assignment_sha256)
    uvicorn.run(app, host=args.listen_host, port=args.port, workers=1, reload=False,
                lifespan="on", interface="asgi3", loop="asyncio", http="h11",
                ws="none", proxy_headers=False, access_log=False,
                limit_concurrency=32, backlog=32)


if __name__ == "__main__":
    main()
