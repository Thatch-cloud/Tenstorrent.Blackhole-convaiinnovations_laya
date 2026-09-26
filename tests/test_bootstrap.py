from dataclasses import asdict
import hashlib
import json
import time

import pytest

from laya_tt.admission import AdmissionRejected, PreparedInput
from laya_tt.bootstrap import RuntimeApplication
from laya_tt.ledger import ExecutionJournal, JournalConflict
from laya_tt.ledger_client import LedgerUnavailable, LocalLedgerClient
from laya_tt.readback import BackendReadback, ReadbackUnavailable
from laya_tt.service import RuntimeIdentity, ServiceNotReady


RUNTIME = RuntimeIdentity("a" * 40, "build", "generation", "cpu-reference")


class Backend:
    def __init__(self):
        self.executions = 0
        self.mode = "cpu-reference"
        self.application = None

    def readback(self):
        return BackendReadback(self.mode, "float32", ("choice", "score", "noul"),
                               2, 64, 512, 192, 1024)

    def prepare(self, request):
        return PreparedInput(1, 5, request["state"])

    def execute(self, _state):
        self.executions += 1
        assert not self.application.journal.pending_consumptions()
        assert self.application.journal.unresolved()[0]["state"] == "running"
        return {"answers": {"q": {"type": "noul", "noul": 0.5,
                "confidence": 1., "answer_confidence": 1.,
                "action": {"act_probability": 0.5}}}, "encoded_tokens": 5}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    raw = json.dumps({"model": "laya-english", "state": "fixture",
        "questions": {"q": {"type": "noul", "instructions": "Evaluate"}}}).encode()
    now = int(time.time() * 1000)
    context = dict(schema_version="1", request_id="request", execution_attempt_id="attempt",
        usage_event_id="usage", tenant_id="tenant", key_id="key", runtime_generation="generation",
        policy_revision="policy", reservation_id="reservation", model="laya-english",
        checkpoint_revision="a" * 40, issued_at_unix_ms=now - 1000, deadline_unix_ms=now + 60000,
        max_question_rows=1, max_encoded_tokens=20, request_sha256=hashlib.sha256(raw).hexdigest())

    def verifier(envelope):
        if envelope != b"authenticated-test-fixture":
            raise PermissionError()
        return context

    backend = Backend()
    app = RuntimeApplication(backend=backend, runtime=RUNTIME, verify_grant=verifier,
        policy_revision="policy", journal_path=tmp_path / "journal.sqlite",
        ledger_socket=tmp_path / "ledger.sock", ledger_uid=1000)
    backend.application = app
    requests = []

    def exchange(_client, route, body, headers, expected_status, deadline):
        requests.append((route, body))
        if route.endswith("consume"):
            assert app.journal.pending_consumptions()[0]["state"] == "pending"
            assert backend.executions == 0
            return b""
        return json.dumps({"receipt_id": headers["x-thatch-receipt-id"],
                           "payload_sha256": headers["x-thatch-receipt-sha256"]}).encode()

    monkeypatch.setattr(LocalLedgerClient, "_exchange", exchange)
    try:
        yield app, backend, raw, context, requests
    finally:
        app.drain(timeout=2, cancel_queued=True)


def test_composition_requires_start_then_journals_before_execution_and_acknowledges(setup):
    app, backend, raw, context, requests = setup
    with pytest.raises(ServiceNotReady):
        app.service.submit(raw, b"authenticated-test-fixture")
    assert not requests
    app.start(lambda: True)  # Test backend; never evidence of a real model self-test.
    response = app.service.submit(raw, b"authenticated-test-fixture").result(2)
    assert response["usage"]["encoded_tokens"] == 5
    assert backend.executions == 1
    assert app.journal.pending_consumptions() == []
    assert len(app.journal.pending_receipts()) == 1
    assert app.deliver_pending() == 1
    assert app.journal.pending_receipts() == []
    assert [route for route, _body in requests] == ["/v1/ledger/consume", "/v1/ledger/receipts"]


def test_rejected_grant_never_consumes_or_executes(setup):
    app, backend, raw, context, requests = setup
    app.start(lambda: True)
    with pytest.raises(AdmissionRejected):
        app.service.submit(raw, b"unauthenticated")
    assert requests == [] and backend.executions == 0
    assert app.journal.pending_consumptions() == []


def test_lost_consumption_ack_keeps_intent_and_never_executes(setup, monkeypatch):
    app, backend, raw, context, requests = setup
    app.start(lambda: True)

    def lost(*args):
        raise OSError("lost reply")

    monkeypatch.setattr(LocalLedgerClient, "_exchange", lost)
    with pytest.raises(AdmissionRejected):
        app.service.submit(raw, b"authenticated-test-fixture")
    assert backend.executions == 0
    assert app.journal.pending_consumptions()[0]["state"] == "pending"
    with pytest.raises(JournalConflict):
        ExecutionJournal(app.journal.path).check_startup()


def test_receipt_failure_preserves_exact_outbox_and_releases_pump_lock(setup, monkeypatch):
    app, backend, raw, context, requests = setup
    app.start(lambda: True)
    app.service.submit(raw, b"authenticated-test-fixture").result(2)
    original = app.journal.pending_receipts()
    monkeypatch.setattr(LocalLedgerClient, "_exchange", lambda *args: b"{}")
    for _ in range(2):
        with pytest.raises(LedgerUnavailable, match="receipt not acknowledged"):
            app.deliver_pending()
        assert app.journal.pending_receipts() == original


def test_changed_loaded_backend_during_self_test_stays_unready(setup):
    app, backend, raw, context, requests = setup

    def self_test():
        backend.mode = "tt-blackhole"
        return True

    with pytest.raises(ReadbackUnavailable):
        app.start(self_test)
    with pytest.raises(ServiceNotReady):
        app.service.submit(raw, b"authenticated-test-fixture")
    assert requests == []


def test_existing_intent_prevents_composition(tmp_path):
    from test_ledger_client import claims
    journal = ExecutionJournal(tmp_path / "journal.sqlite")
    journal.begin_consumption(claims(), asdict(RUNTIME), 2, 17)
    with pytest.raises(JournalConflict):
        RuntimeApplication(backend=Backend(), runtime=RUNTIME, verify_grant=lambda _: {},
            policy_revision="policy", journal_path=journal.path,
            ledger_socket=tmp_path / "ledger.sock", ledger_uid=1000)

def test_hosted_lifespan_drives_actual_runtime_readiness_and_drain(setup):
    import asyncio
    from laya_tt.hosting import HostedRuntime

    app, backend, raw, context, requests = setup
    host = HostedRuntime(app, self_test=lambda: True, receipt_interval=.01)

    async def run():
        incoming = asyncio.Queue()
        await incoming.put({"type": "lifespan.startup"})
        messages = []

        async def send(message):
            messages.append(message)
            if message["type"] == "lifespan.startup.complete":
                assert app.service.readiness()["ready"] is True
                await incoming.put({"type": "lifespan.shutdown"})

        await host({"type": "lifespan"}, incoming.get, send)
        assert [m["type"] for m in messages] == [
            "lifespan.startup.complete", "lifespan.shutdown.complete"]
        assert app.service.readiness()["state"] == "stopped"
    asyncio.run(run())
