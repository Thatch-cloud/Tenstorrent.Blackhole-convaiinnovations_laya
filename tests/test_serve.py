from dataclasses import asdict
import asyncio
import hashlib
import json
from pathlib import Path
import socket

import pytest

from laya_tt.admission import PreparedInput
from laya_tt.readback import BackendReadback
from laya_tt.service import RuntimeIdentity, ServiceNotReady
from laya_tt import serve

ROOT = Path(__file__).resolve().parents[1]


def test_assignment_digest_checks_exact_bytes_before_parsing_or_model_load(tmp_path, monkeypatch):
    path = tmp_path / "assignment.json"
    raw = b'{"identity":"original"}'
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    assert serve._read_json(path, expected_sha256=digest) == {"identity": "original"}
    path.write_bytes(raw + b"\n")
    monkeypatch.setattr(serve, "load_cpu_backend", lambda *a, **k: pytest.fail("model loaded"))
    with pytest.raises(ValueError, match="pinned digest"):
        serve.build_cpu_runtime(path, root=ROOT, assignment_sha256=digest)


@pytest.fixture
def assignment(tmp_path):
    value = dict(schema_version=1,
        runtime=asdict(RuntimeIdentity("55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
                                     "test-build", "test-generation", "cpu-reference")),
        issuer="test-issuer", host_id="test-host", policy_revision="test-policy",
        public_keys={"test-key": "12" * 32}, journal_path=str(tmp_path / "journal.sqlite"),
        ledger_socket=str(tmp_path / "ledger.sock"), ledger_uid=1000)
    path = tmp_path / "assignment.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, value


@pytest.mark.parametrize("mutate", [
    lambda v: v.update(unknown=True),
    lambda v: v.update(schema_version=True),
    lambda v: v.update(ledger_uid=True),
    lambda v: v.update(journal_path="relative.sqlite"),
    lambda v: v.update(public_keys={"test-key": "secret"}),
    lambda v: v["runtime"].update(backend="tt-blackhole"),
    lambda v: v["runtime"].update(checkpoint_revision="a" * 40),
])
def test_invalid_assignment_never_loads_model(assignment, monkeypatch, mutate):
    path, value = assignment
    mutate(value)
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(serve, "load_cpu_backend", lambda *a, **k: pytest.fail("model loaded"))
    with pytest.raises(ValueError):
        serve.build_cpu_runtime(path, root=ROOT)


def test_duplicate_or_oversized_assignment_never_loads_model(assignment, monkeypatch):
    path, _ = assignment
    monkeypatch.setattr(serve, "load_cpu_backend", lambda *a, **k: pytest.fail("model loaded"))
    for raw in ('{"schema_version":1,"schema_version":1}', " " * 65537):
        path.write_text(raw, encoding="utf-8")
        with pytest.raises(ValueError):
            serve.build_cpu_runtime(path, root=ROOT)


@pytest.mark.parametrize("correct", [True, False])
def test_startup_requires_execution_and_pinned_answers(assignment, monkeypatch, correct):
    path, _ = assignment
    reference_path = ROOT / "tests/fixtures/cpu-reference/reference.json"
    reference = json.loads(reference_path.read_bytes())
    case = next(c for c in reference["cases"] if c["id"] == "mixed-question-widths")
    expected = json.loads((reference_path.parent / case["answers"]["file"]).read_bytes())[0]["answers"]

    class Backend:
        executions = 0

        def readback(self):
            return BackendReadback("cpu-reference", "float32", ("choice", "score", "noul"),
                                   64, 64, 512, 192, 32768)

        def prepare(self, request):
            assert request["questions"] == case["input"]["questions"]
            return PreparedInput(4, 157, request)

        def execute(self, _payload):
            self.executions += 1
            return {"answers": expected if correct else {}, "encoded_tokens": 157}

    backend = Backend()
    monkeypatch.setattr(serve, "load_cpu_backend", lambda *a, **k: backend)
    hosted = serve.build_cpu_runtime(path, root=ROOT)
    app = hosted.application
    try:
        assert backend.executions == 0
        assert not app.service.readiness()["ready"]
        if correct:
            import uvicorn

            async def exercise_http():
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    server = uvicorn.Server(uvicorn.Config(hosted, lifespan="on",
                        loop="asyncio", http="h11", ws="none", access_log=False,
                        log_level="error"))
                    task = asyncio.create_task(server.serve(sockets=[listener]))
                    try:
                        async with asyncio.timeout(10):
                            while not server.started:
                                if task.done():
                                    await task
                                    pytest.fail("server stopped before readiness")
                                await asyncio.sleep(.01)
                        assert app.service.readiness()["ready"]
                        reader, writer = await asyncio.open_connection(*listener.getsockname())
                        writer.write(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
                        await writer.drain()
                        response = await asyncio.wait_for(reader.read(), 5)
                        writer.close()
                        await writer.wait_closed()
                        assert response.startswith(b"HTTP/1.1 200")
                        assert b'"cpu-reference"' in response
                    finally:
                        server.should_exit = True
                        await asyncio.wait_for(task, 10)

            asyncio.run(exercise_http())
            assert not app.service.readiness()["ready"]
        else:
            with pytest.raises(ServiceNotReady):
                app.start(hosted.self_test)
            assert not app.service.readiness()["ready"]
        assert backend.executions == 1
        assert app.journal.pending_consumptions() == []
        assert app.journal.pending_receipts() == []
    finally:
        app.drain(timeout=2, cancel_queued=True)
