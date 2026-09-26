from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time

import pytest

from laya_tt.ledger import Receipt
from laya_tt.ledger_client import LedgerUnavailable, LocalLedgerClient
from laya_tt.ledger_protocol import encode_consumption


RUNTIME = dict(model="laya-english", checkpoint_revision="a" * 40,
    runtime_revision="build", runtime_generation="generation", backend="cpu-reference")


def claims():
    now = int(time.time() * 1000)
    return dict(schema_version="1", request_id="request", execution_attempt_id="attempt",
        usage_event_id="usage", tenant_id="tenant", key_id="key", runtime_generation="generation",
        policy_revision="policy", reservation_id="reservation", model="laya-english",
        checkpoint_revision="a" * 40, issued_at_unix_ms=now - 1000, deadline_unix_ms=now + 60000,
        max_question_rows=2, max_encoded_tokens=20, request_sha256="b" * 64)


def receipt():
    value = dict(schema_version="1", outcome="not_started", tenant_id="tenant", request_id="request",
        key_id="key", execution_attempt_id="attempt", reservation_id="reservation", usage_event_id="usage",
        request_sha256="b" * 64, policy_revision="policy", started_at_unix_ms=None,
        occurred_at_unix_ms=2000, runtime=RUNTIME, observation={"reason": "cancelled"})
    payload = json.dumps(value, indent=2) + "\n"
    return Receipt("receipt-1", payload, hashlib.sha256(payload.encode()).hexdigest())


def reply(status=204, body=b"", extra=b""):
    return (f"HTTP/1.1 {status} Result\r\nContent-Length: {len(body)}\r\nConnection: close\r\n".encode()
            + extra + b"\r\n" + body)


def test_owned_runtime_and_validated_counts(monkeypatch, tmp_path):
    observed = dict(RUNTIME)
    client = LocalLedgerClient(tmp_path / "ledger.sock", 1000, observed)
    observed["runtime_generation"] = "mutated"
    context = claims()
    sent = []
    monkeypatch.setattr(client, "_exchange", lambda *args: sent.append(args) or b"")
    assert client.consume_reservation(context, 2, 17) is True
    assert sent[0][1] == encode_consumption(context, RUNTIME, question_rows=2, encoded_tokens=17)
    for context, rows, tokens in [(dict(context, deadline_unix_ms=1), 2, 17),
                                  (context, 3, 17), (context, 2, True)]:
        with pytest.raises(LedgerUnavailable, match="^ledger consumption not acknowledged$"):
            client.consume_reservation(context, rows, tokens)
    assert len(sent) == 1


@pytest.mark.parametrize("bad", ["digest", "id", "runtime", "duplicate", "oversize"])
def test_invalid_receipts_never_connect(monkeypatch, tmp_path, bad):
    client = LocalLedgerClient(tmp_path / "ledger.sock", 1000, RUNTIME)
    evidence = receipt()
    if bad == "digest":
        evidence = replace(evidence, payload_sha256="0" * 64)
    elif bad == "id":
        evidence = replace(evidence, receipt_id="bad\r\nsecret")
    else:
        body = evidence.payload_json
        if bad == "runtime":
            body = body.replace('"generation"', '"other"')
        elif bad == "duplicate":
            body = body.replace('"schema_version": "1",', '"schema_version": "1", "schema_version": "1",')
        else:
            body += " " * 16384
        evidence = replace(evidence, payload_json=body, payload_sha256=hashlib.sha256(body.encode()).hexdigest())
    called = []
    monkeypatch.setattr(client, "_exchange", lambda *args: called.append(args))
    with pytest.raises(LedgerUnavailable, match="^ledger receipt not acknowledged$"):
        client.deliver_receipt(evidence)
    assert not called


LINUX = pytest.mark.skipif(not hasattr(socket, "SO_PEERCRED"), reason="Linux peer credentials required")


@contextmanager
def bridge(tmp_path, response, delay=0):
    # Use a short canonical path: Unix sockaddr path capacity is finite.
    directory = tmp_path.resolve() / "b"
    directory.mkdir(mode=0o700)
    path = directory / "s"
    requests, errors = [], []
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(path))
        path.chmod(0o600)
        listener.listen()
        listener.settimeout(2)

        def serve():
            try:
                with listener.accept()[0] as stream:
                    stream.settimeout(2)
                    raw = b""
                    while b"\r\n\r\n" not in raw:
                        block = stream.recv(1024)
                        if not block:
                            break
                        raw += block
                    if raw:
                        head, body = raw.split(b"\r\n\r\n", 1)
                        size = int(next(line.split(b":", 1)[1] for line in head.split(b"\r\n")
                                        if line.lower().startswith(b"content-length:")))
                        while len(body) < size:
                            block = stream.recv(size - len(body))
                            if not block:
                                raise ValueError("truncated request")
                            body += block
                        raw = head + b"\r\n\r\n" + body
                    requests.append(raw)
                    time.sleep(delay)
                    if response is not None:
                        try:
                            stream.sendall(response)
                        except (BrokenPipeError, ConnectionResetError):
                            pass
                listener.settimeout(0.1)
                try:
                    repeated, _address = listener.accept()
                except socket.timeout:
                    pass
                else:
                    repeated.close()
                    raise AssertionError("client retried delivery")
            except Exception as exc:
                errors.append(exc)

        task = threading.Thread(target=serve, daemon=True)
        task.start()
        try:
            yield path, requests
        finally:
            task.join(3)
            assert not task.is_alive()
            assert not errors, errors


@LINUX
def test_real_peer_and_exact_consumption_bytes(tmp_path):
    context = claims()
    with bridge(tmp_path, reply()) as (path, requests):
        client = LocalLedgerClient(path, os.getuid(), RUNTIME)
        assert client.consume_reservation(context, 2, 17) is True
    assert len(requests) == 1
    assert requests[0].startswith(b"POST /v1/ledger/consume HTTP/1.1\r\n")
    assert requests[0].endswith(encode_consumption(context, RUNTIME, question_rows=2, encoded_tokens=17))
    assert b"authorization" not in requests[0].lower()


@LINUX
@pytest.mark.parametrize("response", [None, reply(302, extra=b"Location: http://127.0.0.1/\r\n"),
    reply(409), reply(503), reply(200), reply(204, b"x"),
    reply(extra=b"Transfer-Encoding: chunked\r\n"),
    reply(extra=b"Content-Length: 0\r\n"), b"HTTP/1.1 204 OK\r\nX: " + b"a" * 8192])
def test_consumption_failure_is_single_attempt(tmp_path, response):
    with bridge(tmp_path, response) as (path, requests):
        with pytest.raises(LedgerUnavailable):
            LocalLedgerClient(path, os.getuid(), RUNTIME).consume_reservation(claims(), 2, 17)
    assert len(requests) == 1


@LINUX
@pytest.mark.parametrize("ack_kind", ["exact", "wrong_id", "wrong_digest", "extra", "duplicate", "oversize"])
def test_receipt_original_bytes_and_acknowledgment(tmp_path, ack_kind):
    evidence = receipt()
    ack = {"receipt_id": evidence.receipt_id, "payload_sha256": evidence.payload_sha256}
    if ack_kind == "wrong_id":
        ack["receipt_id"] = "other"
    if ack_kind == "wrong_digest":
        ack["payload_sha256"] = "0" * 64
    if ack_kind == "extra":
        ack["extra"] = True
    raw = json.dumps(ack).encode()
    if ack_kind == "duplicate":
        raw = raw.replace(b'{', b'{"receipt_id":"receipt-1",', 1)
    if ack_kind == "oversize":
        raw += b" " * 4096
    with bridge(tmp_path, reply(200, raw)) as (path, requests):
        client = LocalLedgerClient(path, os.getuid(), RUNTIME)
        if ack_kind == "exact":
            assert client.deliver_receipt(evidence) is True
        else:
            with pytest.raises(LedgerUnavailable):
                client.deliver_receipt(evidence)
    assert requests[0].endswith(evidence.payload_json.encode())
    assert b"x-thatch-receipt-id: receipt-1\r\n" in requests[0]


@LINUX
@pytest.mark.parametrize("unsafe", ["directory", "socket", "symlink", "uid"])
def test_unsafe_endpoint_rejected_without_send(tmp_path, unsafe):
    directory = tmp_path.resolve() / "private"
    directory.mkdir(mode=0o700)
    path = directory / "s"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(str(path))
        listener.listen()
        listener.settimeout(0.05)
        path.chmod(0o600)
        uid = os.getuid()
        if unsafe == "directory":
            directory.chmod(0o755)
        elif unsafe == "socket":
            path.chmod(0o666)
        elif unsafe == "uid":
            uid += 1
        else:
            alias = tmp_path / "alias"
            alias.symlink_to(directory, target_is_directory=True)
            path = alias / "s"
        with pytest.raises(LedgerUnavailable):
            LocalLedgerClient(path, uid, RUNTIME).consume_reservation(claims(), 2, 17)
        with pytest.raises(socket.timeout):
            listener.accept()


@LINUX
def test_operation_deadline_includes_response_wait(tmp_path):
    context = claims()
    with bridge(tmp_path, reply(), delay=0.3) as (path, requests):
        context["deadline_unix_ms"] = int(time.time() * 1000) + 100
        with pytest.raises(LedgerUnavailable):
            LocalLedgerClient(path, os.getuid(), RUNTIME).consume_reservation(context, 2, 17)
    assert len(requests) == 1


@LINUX
def test_kernel_server_uid_checked_before_sending(monkeypatch, tmp_path):
    import struct
    original = socket.socket

    class WrongPeer(original):
        def getsockopt(self, level, option, *args):
            if level == socket.SOL_SOCKET and option == socket.SO_PEERCRED:
                return struct.pack("3i", os.getpid(), os.getuid() + 1, os.getgid())
            return super().getsockopt(level, option, *args)

    with bridge(tmp_path, None) as (path, requests):
        monkeypatch.setattr(socket, "socket", WrongPeer)
        with pytest.raises(LedgerUnavailable):
            LocalLedgerClient(path, os.getuid(), RUNTIME).consume_reservation(claims(), 2, 17)
    assert requests == [b""]


def test_unsupported_peer_authentication_fails_closed(monkeypatch, tmp_path):
    monkeypatch.delattr(socket, "SO_PEERCRED", raising=False)
    client = LocalLedgerClient(tmp_path / "unused.sock", 1000, RUNTIME)
    with pytest.raises(LedgerUnavailable):
        client.consume_reservation(claims(), 2, 17)


def test_journal_intent_precedes_network_and_survives_uncertain_response(monkeypatch, tmp_path):
    from laya_tt.ledger import ExecutionJournal, JournalConflict, JournalReplay
    journal = ExecutionJournal(tmp_path / "journal.sqlite")
    client = LocalLedgerClient(tmp_path / "unused.sock", 1000, RUNTIME)
    context = claims()
    calls = []

    def uncertain(*args):
        calls.append(args)
        reopened = ExecutionJournal(journal.path)
        assert reopened.pending_consumptions()[0]["state"] == "pending"
        with pytest.raises(JournalConflict):
            reopened.check_startup()
        raise LedgerUnavailable("lost acknowledgment")

    monkeypatch.setattr(client, "consume_reservation", uncertain)
    consume = client.journaled_consumption(journal)
    with pytest.raises(LedgerUnavailable):
        consume(context, 2, 17)
    with pytest.raises(JournalReplay):
        consume(context, 2, 17)
    assert len(calls) == 1
    assert journal.pending_receipts() == []
    with pytest.raises(JournalConflict):
        journal.admitted(context, 2, 17, RUNTIME)
    assert journal.pending_consumptions()[0]["state"] == "pending"


def test_acknowledged_intent_moves_atomically_to_matching_admission(monkeypatch, tmp_path):
    from laya_tt.ledger import ExecutionJournal, JournalConflict, JournalReplay
    journal = ExecutionJournal(tmp_path / "journal.sqlite")
    client = LocalLedgerClient(tmp_path / "unused.sock", 1000, RUNTIME)
    monkeypatch.setattr(client, "consume_reservation", lambda *args: True)
    context = claims()
    assert client.journaled_consumption(journal)(context, 2, 17) is True
    reopened = ExecutionJournal(journal.path)
    assert reopened.pending_consumptions()[0]["state"] == "acknowledged"
    with pytest.raises(JournalConflict):
        reopened.check_startup()
    with pytest.raises(JournalConflict):
        reopened.admitted(context, 2, 18, RUNTIME)
    assert reopened.pending_consumptions()[0]["state"] == "acknowledged"
    reopened.admitted(context, 2, 17, RUNTIME)
    assert reopened.pending_consumptions() == []
    assert reopened.status(context)["state"] == "admitted"
    with pytest.raises(JournalReplay):
        client.journaled_consumption(reopened)(context, 2, 17)


def test_intent_identity_is_unique_across_connections(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from laya_tt.ledger import ExecutionJournal, JournalReplay
    path = tmp_path / "journal.sqlite"
    first, second = ExecutionJournal(path), ExecutionJournal(path)
    context = claims()

    def attempt(journal):
        try:
            journal.begin_consumption(context, RUNTIME, 2, 17)
            return True
        except JournalReplay:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(attempt, [first, second])) == [False, True]
    for change in ({"execution_attempt_id": "different"},
                   {"reservation_id": "different"}, {"usage_event_id": "different"}):
        with pytest.raises(JournalReplay):
            first.begin_consumption(dict(context, **change), RUNTIME, 2, 17)
    assert len(first.pending_consumptions()) == 1


def test_version_one_upgrade_preserves_execution_evidence(tmp_path):
    import sqlite3
    from laya_tt.ledger import ExecutionJournal
    path = tmp_path / "journal.sqlite"
    journal = ExecutionJournal(path)
    context = claims()
    journal.admitted(context, 2, 17, RUNTIME)
    journal.not_started(context, "cancelled")
    original = journal.pending_receipts()
    # Reproduce the previous schema: no intent table and user_version=1.
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE consumption_intents")
        db.execute("PRAGMA user_version=1")
    reopened = ExecutionJournal(path)
    assert reopened.pending_receipts() == original
    assert reopened.pending_consumptions() == []
    reopened.check_startup()
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 2
