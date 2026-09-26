"""Opt-in real CPU model through runtime, signed admission and Unix ledger I/O.

The ledger peer is a recording fixture, NOT Administration or a quota authority.
This closes the Python runtime composition boundary, not platform/hardware G2/G3.
"""
import asyncio
import base64
from dataclasses import asdict
import hashlib
import json
import os
import signal
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from laya_tt.bootstrap import RuntimeApplication
from laya_tt.cpu_backend import load_cpu_backend
from laya_tt.grants import DecisionGrantVerifier
from laya_tt.hosting import HostedRuntime
from laya_tt.ledger import ExecutionJournal
from laya_tt.ledger_protocol import encode_consumption
from laya_tt.service import RuntimeIdentity


def signed(key, runtime, context):
    def encode(value):
        raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=")
    prefix = encode({"alg": "Ed25519", "typ": "thatch-decision-grant+jws", "kid": "test-key"})
    prefix += b"." + encode({"schema_version": "1", "issuer": "test-authority", "host_id": "test-host",
                             "runtime": runtime, "context": context})
    return prefix + b"." + base64.urlsafe_b64encode(key.sign(prefix)).rstrip(b"=")


async def invoke(application, raw, grant):
    incoming = asyncio.Queue()
    await incoming.put({"type": "http.request", "body": raw})
    outgoing = []
    async def send(message):
        outgoing.append(message)
    scope = {"type": "http", "path": "/v1/decisions", "method": "POST",
             "headers": [(b"content-type", b"application/json"),
                         (b"x-thatch-admission-grant", base64.b64encode(grant))]}
    await asyncio.wait_for(application(scope, incoming.get, send), 60)
    return outgoing


@unittest.skipUnless(os.environ.get("LAYA_CPU_INTEGRATION") == "1" and hasattr(socket, "SO_PEERCRED"),
                     "requires explicit CPU integration, pinned assets and Linux peer credentials")
class CpuRuntimeIntegrationTests(unittest.TestCase):
    def setUp(self):
        if os.environ.get("LAYA_CPU_EXPECT_UID"):
            self.assertEqual(os.getuid(), int(os.environ["LAYA_CPU_EXPECT_UID"]))

    def test_cpu_entrypoint_real_model_inventory_and_signal_shutdown(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="laya-entrypoint-") as temporary:
            directory = Path(temporary).resolve()
            runtime = RuntimeIdentity("55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
                                      "cpu-entrypoint-test", "entrypoint-generation", "cpu-reference")
            key = Ed25519PrivateKey.generate()
            assignment = dict(schema_version=1, runtime=asdict(runtime), issuer="test-authority",
                host_id="test-host", policy_revision="test-policy",
                public_keys={"test-key": key.public_key().public_bytes_raw().hex()},
                journal_path=str(directory / "journal.sqlite"),
                ledger_socket=str(directory / "ledger.sock"), ledger_uid=os.getuid())
            path = directory / "assignment.json"
            path.write_text(json.dumps(assignment), encoding="utf-8")
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            with (directory / "process.log").open("w+", encoding="utf-8") as log:
                process = subprocess.Popen([sys.executable, "-m", "laya_tt.serve",
                    "--assignment", str(path), "--root", str(root), "--port", str(port),
                    "--listen-host", os.environ.get("LAYA_CPU_TEST_LISTEN_HOST", "127.0.0.1")],
                    stdout=log, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic() + 120
                    inventory = None
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            log.seek(0)
                            self.fail("entrypoint stopped before readiness: " + log.read())
                        try:
                            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1) as response:
                                inventory = json.load(response)
                            break
                        except (OSError, ValueError):
                            time.sleep(.1)
                    self.assertIsNotNone(inventory, "entrypoint did not become ready")
                    self.assertIn("entrypoint-generation", json.dumps(inventory))
                    self.assertIn('"state": "ready"', json.dumps(inventory))
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=40)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                        self.fail("entrypoint did not drain on SIGTERM")
                # Uvicorn restores and re-raises SIGTERM after graceful shutdown.
                self.assertEqual(process.returncode, -signal.SIGTERM)
                log.seek(0)
                diagnostics = log.read()
                self.assertIn("Application shutdown complete", diagnostics)
                self.assertIn("http://" + os.environ.get("LAYA_CPU_TEST_LISTEN_HOST", "127.0.0.1") + ":", diagnostics)

    def test_pinned_answers_signed_admission_real_peer_and_durable_outbox(self):
        self.exercise(False)

    def test_hosted_lifespan_real_model_and_automatic_receipt_delivery(self):
        self.exercise(True)

    def exercise(self, hosted):
        root = Path(__file__).resolve().parents[1]
        reference_path = root / "tests/fixtures/cpu-reference/reference.json"
        raw_reference = reference_path.read_bytes()
        self.assertEqual(hashlib.sha256(raw_reference).hexdigest(),
                         "d1cb5aaaebdc21b5b8c6db1287811fb098659370978e9d1f50e593b329acb085")
        reference = json.loads(raw_reference)
        case = next(c for c in reference["cases"] if c["id"] == "mixed-question-widths")
        expected_bytes = (reference_path.parent / case["answers"]["file"]).read_bytes()
        self.assertEqual(hashlib.sha256(expected_bytes).hexdigest(), case["answers"]["sha256"])
        expected = json.loads(expected_bytes)[0]["answers"]
        request = {"model": "laya-english", "state": case["input"]["state"],
                   "questions": case["input"]["questions"]}
        raw = json.dumps(request, ensure_ascii=False).encode()
        backend = load_cpu_backend(root / "configs/checkpoint-lock.json", root=root)
        prepared = backend.prepare(request)
        runtime = RuntimeIdentity(reference["manifest"]["checkpoint"]["revision"],
                                  "cpu-runtime-integration", "test-generation", "cpu-reference")
        key = Ed25519PrivateKey.generate()
        verifier = DecisionGrantVerifier(public_keys={"test-key": key.public_key().public_bytes_raw()},
            issuer="test-authority", host_id="test-host", runtime=asdict(runtime), policy_revision="test-policy")
        with tempfile.TemporaryDirectory(prefix="laya-runtime-") as temporary:
            directory = Path(temporary).resolve()
            directory.chmod(0o700)
            path = directory / "ledger.sock"
            app = RuntimeApplication(backend=backend, runtime=runtime, verify_grant=verifier,
                policy_revision="test-policy", journal_path=directory / "journal.sqlite",
                ledger_socket=path, ledger_uid=os.getuid())
            try:
                # Startup executes the loaded model and checks committed golden
                # answers for choice, score, noul and a one-option question.
                def self_test():
                    output = backend.execute(prepared.payload)
                    return output["answers"] == expected and output["encoded_tokens"] == prepared.encoded_tokens
                if not hosted:
                    app.start(self_test)
                    self.assertTrue(app.service.readiness()["ready"])
                now = int(time.time() * 1000)
                context = dict(schema_version="1", request_id="request", execution_attempt_id="attempt",
                    usage_event_id="usage", tenant_id="tenant", key_id="key", runtime_generation=runtime.runtime_generation,
                    policy_revision="test-policy", reservation_id="reservation", model=runtime.model,
                    checkpoint_revision=runtime.checkpoint_revision, issued_at_unix_ms=now, deadline_unix_ms=now+60000,
                    max_question_rows=prepared.question_rows, max_encoded_tokens=prepared.encoded_tokens,
                    request_sha256=hashlib.sha256(raw).hexdigest())
                seen, errors, retained = [], [], []
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                    listener.bind(str(path))
                    path.chmod(0o600)
                    listener.listen(2)
                    listener.settimeout(10)
                    def peer():
                        try:
                            for index in range(2):
                                with listener.accept()[0] as stream:
                                    stream.settimeout(10)
                                    data = b""
                                    while b"\r\n\r\n" not in data:
                                        block = stream.recv(1024)
                                        if not block or len(data) > 8192:
                                            raise AssertionError("invalid fixture request headers")
                                        data += block
                                    head, body = data.split(b"\r\n\r\n", 1)
                                    lines = head.decode("ascii").split("\r\n")
                                    headers = {name.lower(): value for name, value in
                                               (line.split(": ", 1) for line in lines[1:])}
                                    size = int(headers["content-length"])
                                    self.assertLessEqual(size, 16384)
                                    while len(body) < size:
                                        block = stream.recv(size-len(body))
                                        if not block:
                                            raise AssertionError("truncated fixture request")
                                        body += block
                                    seen.append((lines[0], body))
                                    self.assertNotIn("authorization", headers)
                                    if index == 0:
                                        self.assertEqual(lines[0], "POST /v1/ledger/consume HTTP/1.1")
                                        self.assertEqual(body, encode_consumption(context, asdict(runtime),
                                            question_rows=prepared.question_rows, encoded_tokens=prepared.encoded_tokens))
                                        self.assertEqual(len(app.journal.pending_consumptions()), 1)
                                        status, response = 204, b""
                                    else:
                                        self.assertEqual(lines[0], "POST /v1/ledger/receipts HTTP/1.1")
                                        self.assertEqual(hashlib.sha256(body).hexdigest(), headers["x-thatch-receipt-sha256"])
                                        pending = ExecutionJournal(app.journal.path).pending_receipts()
                                        self.assertEqual(len(pending), 1)
                                        self.assertEqual(pending[0].payload_json.encode(), body)
                                        retained.append(pending[0])
                                        status, response = 200, json.dumps({"receipt_id": headers["x-thatch-receipt-id"],
                                            "payload_sha256": headers["x-thatch-receipt-sha256"]}).encode()
                                    stream.sendall(f"HTTP/1.1 {status} Result\r\nContent-Length: {len(response)}\r\nConnection: close\r\n\r\n".encode()+response)
                        except BaseException as error:
                            errors.append(error)
                    thread = threading.Thread(target=peer, daemon=True)
                    if not hosted:
                        thread.start()
                    try:
                        grant = signed(key, asdict(runtime), context)
                        async def through_lifespan():
                            host = HostedRuntime(app, self_test=self_test, receipt_interval=.01)
                            events, started = asyncio.Queue(), asyncio.Event()
                            lifecycle_messages = []
                            async def send(message):
                                lifecycle_messages.append(message)
                                started.set()
                            await events.put({"type": "lifespan.startup"})
                            task = asyncio.create_task(host({"type": "lifespan"}, events.get, send))
                            try:
                                await asyncio.wait_for(started.wait(), 60)
                                self.assertEqual(lifecycle_messages[0]["type"], "lifespan.startup.complete")
                                self.assertTrue(app.service.readiness()["ready"])
                                # Grant lifetime starts after the real startup self-test.
                                issued = int(time.time() * 1000)
                                context.update(issued_at_unix_ms=issued, deadline_unix_ms=issued+60000)
                                thread.start()
                                response = await invoke(host, raw, signed(key, asdict(runtime), context))
                                async def acknowledged():
                                    while not retained or ExecutionJournal(app.journal.path).pending_receipts():
                                        await asyncio.sleep(.01)
                                await asyncio.wait_for(acknowledged(), 10)
                                await events.put({"type": "lifespan.shutdown"})
                                await asyncio.wait_for(task, 30)
                                self.assertEqual(lifecycle_messages[-1]["type"], "lifespan.shutdown.complete")
                                self.assertEqual(app.service.readiness()["state"], "stopped")
                                return response
                            finally:
                                if not task.done():
                                    task.cancel()
                                    await asyncio.gather(task, return_exceptions=True)
                        messages = asyncio.run(through_lifespan() if hosted else invoke(app.asgi, raw, grant))
                        self.assertEqual(messages[0]["status"], 200, messages)
                        response = json.loads(messages[1]["body"])
                        self.assertEqual(response["answers"], expected)
                        self.assertEqual(list(response["answers"]), list(request["questions"]))
                        self.assertEqual(response["usage"]["encoded_tokens"], prepared.encoded_tokens)
                        self.assertEqual(response["request_id"], context["request_id"])
                        self.assertEqual(app.journal.pending_consumptions(), [])
                        if not hosted:
                            pending = app.journal.pending_receipts()
                            self.assertEqual(len(pending), 1)
                            self.assertEqual(ExecutionJournal(app.journal.path).pending_receipts(), pending)
                            self.assertEqual(app.deliver_pending(), 1)
                        self.assertEqual(ExecutionJournal(app.journal.path).pending_receipts(), [])
                    finally:
                        if thread.ident is not None:
                            thread.join(11)
                    self.assertFalse(thread.is_alive())
                    self.assertEqual(errors, [])
                    self.assertEqual(len(retained), 1)
                    self.assertEqual(seen[1][1], retained[0].payload_json.encode())
                    receipt = json.loads(seen[1][1])
                    self.assertEqual(receipt["outcome"], "completed")
                    self.assertEqual(receipt["observation"]["question_rows"], prepared.question_rows)
                    self.assertEqual(receipt["observation"]["encoded_tokens"], prepared.encoded_tokens)
                    self.assertEqual(receipt["runtime"], asdict(runtime))
            finally:
                self.assertTrue(app.drain(timeout=10, cancel_queued=True))


if __name__ == "__main__":
    unittest.main()
