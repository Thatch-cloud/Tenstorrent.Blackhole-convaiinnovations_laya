import hashlib
import json
import sqlite3

import pytest

from laya_tt.ledger import ExecutionJournal, JournalConflict, JournalReplay
from laya_tt.ledger_client import LocalLedgerClient, LedgerUnavailable
from test_ledger_client import RUNTIME, claims, bridge, reply, LINUX


def setup(tmp_path):
    journal = ExecutionJournal(tmp_path / "journal.db")
    context = claims()
    context.update(issued_at_unix_ms=1, deadline_unix_ms=2)
    body = journal.begin_consumption(context, RUNTIME, 2, 17)
    return journal, context, body


def ack(body, outcome):
    return json.dumps(dict(schema_version="1", request_body_sha256=hashlib.sha256(body).hexdigest(),
                           outcome=outcome)).encode()


def test_fence_retains_replay_identity_and_allows_restart(tmp_path, monkeypatch):
    journal, context, body = setup(tmp_path)
    client = LocalLedgerClient(tmp_path / "s", 1000, RUNTIME)
    sent = []
    monkeypatch.setattr(client, "_exchange", lambda *args: sent.append(args) or
                        ack(body, {"state": "unconsumed_fenced"}))
    assert client.recover_pending(journal)[0]["outcome"]["state"] == "unconsumed_fenced"
    assert sent[0][0:2] == ("/v1/ledger/recover", body)
    restarted = ExecutionJournal(journal.path)
    restarted.check_startup()
    assert restarted.pending_consumptions() == []
    with pytest.raises(JournalReplay):
        restarted.begin_consumption(context, RUNTIME, 2, 17)
    with pytest.raises(JournalConflict):
        restarted.admitted(context, 2, 17, RUNTIME)
    for key in ("reservation_id", "usage_event_id"):
        changed = dict(context, execution_attempt_id="new", reservation_id="new", usage_event_id="new")
        changed[key] = context[key]
        with pytest.raises(JournalReplay):
            restarted.begin_consumption(changed, RUNTIME, 2, 17)


@pytest.mark.parametrize("outcome", [
    {"state": "consumed", "prepared_question_rows": 2, "prepared_encoded_tokens": 17},
    {"state": "settled", "receipt_id": "receipt", "payload_sha256": "a" * 64},
])
def test_consumed_and_settled_remain_blocked(tmp_path, monkeypatch, outcome):
    journal, context, body = setup(tmp_path)
    client = LocalLedgerClient(tmp_path / "s", 1000, RUNTIME)
    monkeypatch.setattr(client, "_exchange", lambda *args: ack(body, outcome))
    assert client.recover_pending(journal)[0]["outcome"] == outcome
    assert journal.pending_consumptions()[0]["state"] == "pending"
    with pytest.raises(JournalConflict):
        journal.check_startup()


@pytest.mark.parametrize("bad", ["hash", "duplicate", "array", "extra", "boolean", "ceiling", "lost", "acknowledged", "runtime"])
def test_invalid_recovery_retains_intent(tmp_path, monkeypatch, bad):
    journal, context, body = setup(tmp_path)
    client = LocalLedgerClient(tmp_path / "s", 1000,
        dict(RUNTIME, runtime_generation="wrong") if bad == "runtime" else RUNTIME)
    response = ack(body, {"state": "unconsumed_fenced"})
    if bad == "hash":
        response = ack(b"wrong", {"state": "unconsumed_fenced"})
    elif bad == "duplicate":
        response = response.replace(b'"state":', b'"state":"unconsumed_fenced","state":')
    elif bad == "array":
        response = b"[]"
    elif bad == "extra":
        response = ack(body, {"state": "unconsumed_fenced", "extra": 1})
    elif bad in ("boolean", "ceiling"):
        response = ack(body, {"state": "consumed", "prepared_question_rows": True if bad == "boolean" else 3,
                              "prepared_encoded_tokens": 17})
    elif bad == "lost":
        response = b""
    elif bad == "acknowledged":
        journal.consumption_acknowledged(context, body)
    sent = []
    monkeypatch.setattr(client, "_exchange", lambda *args: sent.append(args) or response)
    with pytest.raises(LedgerUnavailable):
        client.recover_pending(journal)
    assert len(sent) == (0 if bad == "runtime" else 1)
    assert len(journal.pending_consumptions()) == 1
    with pytest.raises(JournalConflict):
        journal.check_startup()


def test_fence_cas_rejects_changed_evidence(tmp_path):
    journal, context, body = setup(tmp_path)
    with pytest.raises(JournalConflict):
        journal.confirm_unconsumed_fence(context, body + b" ")
    journal.consumption_acknowledged(context, body)
    with pytest.raises(JournalConflict):
        journal.confirm_unconsumed_fence(context, body)


@pytest.mark.parametrize("version", [1, 2])
def test_migration_preserves_evidence(tmp_path, version):
    journal, context, body = setup(tmp_path)
    other = dict(context, execution_attempt_id="other", reservation_id="other", usage_event_id="other")
    journal.admitted(other, 2, 17, RUNTIME)
    journal.not_started(other, "cancelled")
    receipts = journal.pending_receipts()
    with sqlite3.connect(journal.path) as db:
        if version == 1:
            db.execute("DROP TABLE consumption_intents")
        else:
            db.execute("ALTER TABLE consumption_intents RENAME TO old_intents")
            db.execute("""CREATE TABLE consumption_intents (
                tenant TEXT NOT NULL, attempt TEXT NOT NULL, reservation TEXT NOT NULL,
                usage_event TEXT NOT NULL, context_json TEXT NOT NULL, consumption BLOB NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('pending','acknowledged')),
                PRIMARY KEY(tenant,attempt), UNIQUE(tenant,reservation), UNIQUE(tenant,usage_event))""")
            db.execute("INSERT INTO consumption_intents SELECT * FROM old_intents")
            db.execute("DROP TABLE old_intents")
        db.execute(f"PRAGMA user_version={version}")
    restored = ExecutionJournal(journal.path)
    assert restored.pending_receipts() == receipts
    assert restored.status(other)["state"] == "not_started"
    if version == 2:
        assert restored.pending_consumptions()[0]["consumption"] == body
        restored.confirm_unconsumed_fence(context, body)
    restored.check_startup()
    with sqlite3.connect(journal.path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 3


@LINUX
def test_real_private_socket_recovery(tmp_path):
    import os
    journal, context, body = setup(tmp_path)
    # Preserve a noncanonical original body; recovery must never re-encode it.
    body = json.dumps(json.loads(body), indent=2).encode() + b"\n"
    with sqlite3.connect(journal.path) as db:
        db.execute("UPDATE consumption_intents SET consumption=?", (body,))
    with bridge(tmp_path, reply(200, ack(body, {"state": "unconsumed_fenced"}))) as (path, requests):
        client = LocalLedgerClient(path, os.getuid(), RUNTIME)
        client.recover_pending(journal)
    assert requests[0].endswith(body)
    assert requests[0].startswith(b"POST /v1/ledger/recover HTTP/1.1")
    journal.check_startup()


def test_lost_recovery_reply_can_be_recovered_again(tmp_path, monkeypatch):
    journal, context, body = setup(tmp_path)
    client = LocalLedgerClient(tmp_path / "s", 1000, RUNTIME)
    replies = iter([b"", ack(body, {"state": "unconsumed_fenced"})])
    sent = []
    def exchange(*args):
        sent.append(args)
        return next(replies)
    monkeypatch.setattr(client, "_exchange", exchange)
    with pytest.raises(LedgerUnavailable):
        client.recover_pending(journal)
    client.recover_pending(journal)
    assert [request[:2] for request in sent] == [("/v1/ledger/recover", body)] * 2
    journal.check_startup()


def test_intervening_acknowledgment_cannot_be_cleared(tmp_path, monkeypatch):
    journal, context, body = setup(tmp_path)
    client = LocalLedgerClient(tmp_path / "s", 1000, RUNTIME)
    def exchange(*args):
        ExecutionJournal(journal.path).consumption_acknowledged(context, body)
        return ack(body, {"state": "unconsumed_fenced"})
    monkeypatch.setattr(client, "_exchange", exchange)
    with pytest.raises(LedgerUnavailable):
        client.recover_pending(journal)
    assert journal.pending_consumptions()[0]["state"] == "acknowledged"
