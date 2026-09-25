"""Local durable execution evidence and outbox; not a platform quota authority."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import uuid

from jsonschema import Draft202012Validator
from .contracts import load_schema


class JournalConflict(RuntimeError):
    """Identity, lifecycle or receipt acknowledgment did not match stored evidence."""


class JournalReplay(JournalConflict):
    """A tenant reservation, execution attempt or usage event has already been seen."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Receipt:
    receipt_id: str
    payload_json: str
    payload_sha256: str


class ExecutionJournal:
    """Explicit on-disk SQLite journal with FULL synchronous commits and WAL.

    Inputs must come from authenticated platform admission. This never issues a
    grant, allocates quota, releases an authority reservation or claims billing.
    One local durable database must be shared across this host's runtime workers;
    replay protection does not span unrelated databases or hosts.
    """
    def __init__(self, path: Path):
        if str(path) == ":memory:":
            raise ValueError("execution journal requires an explicit durable path")
        self.path = Path(path).resolve()
        self._validator = Draft202012Validator(load_schema("admission-context"))
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("BEGIN IMMEDIATE")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise JournalConflict("unsupported journal schema version")
            if version == 0:
                if db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchone():
                    raise JournalConflict("refusing to initialize an unrelated database")
                db.execute("""CREATE TABLE executions (
                    tenant TEXT NOT NULL, attempt TEXT NOT NULL,
                    reservation TEXT NOT NULL, usage_event TEXT NOT NULL,
                    context_json TEXT NOT NULL, runtime_json TEXT NOT NULL,
                    question_rows INTEGER NOT NULL, encoded_tokens INTEGER NOT NULL,
                    state TEXT NOT NULL, delivery TEXT NOT NULL DEFAULT 'pending',
                    queue_ms INTEGER, started_at_ms INTEGER, updated_ms INTEGER NOT NULL,
                    PRIMARY KEY(tenant, attempt), UNIQUE(tenant, reservation),
                    UNIQUE(tenant, usage_event))""")
                db.execute("""CREATE TABLE receipts (
                    receipt_id TEXT PRIMARY KEY, tenant TEXT NOT NULL, attempt TEXT NOT NULL,
                    payload_json TEXT NOT NULL, payload_sha256 TEXT NOT NULL,
                    acknowledged INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(tenant, attempt))""")
                db.execute("PRAGMA user_version=1")
            db.commit()

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
        finally:
            if db.in_transaction:
                db.rollback()
            db.close()

    @contextmanager
    def _transaction(self):
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()

    @staticmethod
    def _integer(value, name):
        if type(value) is not int or value < 0:
            raise ValueError(name + " must be a nonnegative integer")

    def admitted(self, context, question_rows, encoded_tokens, runtime):
        context = dict(context)
        self._validator.validate(context)
        if (type(question_rows) is not int or not 1 <= question_rows <= context["max_question_rows"]
                or type(encoded_tokens) is not int or not 1 <= encoded_tokens <= context["max_encoded_tokens"]):
            raise ValueError("admitted usage exceeds verified reservation")
        runtime = dict(runtime)
        if set(runtime) != {"model", "checkpoint_revision", "runtime_revision", "runtime_generation", "backend"}:
            raise ValueError("invalid runtime receipt identity")
        if any(runtime[key] != context[key] for key in ("model", "checkpoint_revision", "runtime_generation")):
            raise JournalConflict("runtime identity differs from verified admission")
        if runtime["backend"] not in ("cpu-reference", "tt-blackhole"):
            raise ValueError("invalid backend identity")
        try:
            with self._transaction() as db:
                db.execute("INSERT INTO executions (tenant,attempt,reservation,usage_event,context_json,runtime_json,question_rows,encoded_tokens,state,updated_ms) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (context["tenant_id"], context["execution_attempt_id"], context["reservation_id"],
                     context["usage_event_id"], _json(context), _json(runtime), question_rows,
                     encoded_tokens, "admitted", int(time.time()*1000)))
        except sqlite3.IntegrityError:
            raise JournalReplay("local reservation, attempt or usage event was already admitted") from None

    def _row(self, db, context):
        row = db.execute("SELECT * FROM executions WHERE tenant=? AND attempt=?",
                         (context["tenant_id"], context["execution_attempt_id"])).fetchone()
        if row is None or row["context_json"] != _json(dict(context)):
            raise JournalConflict("execution identity does not match authenticated admission")
        return row

    def started(self, context, queue_ms):
        self._integer(queue_ms, "queue_ms")
        with self._transaction() as db:
            row = self._row(db, context)
            if row["state"] != "admitted":
                return False
            now = int(time.time()*1000)
            db.execute("UPDATE executions SET state='running',queue_ms=?,started_at_ms=?,updated_ms=? WHERE tenant=? AND attempt=?",
                       (queue_ms, now, now, row["tenant"], row["attempt"]))
        return True

    def _terminal(self, db, row, outcome, observation):
        context, runtime = json.loads(row["context_json"]), json.loads(row["runtime_json"])
        existing = db.execute("SELECT * FROM receipts WHERE tenant=? AND attempt=?",
                              (row["tenant"], row["attempt"])).fetchone()
        occurred_at = (json.loads(existing["payload_json"])["occurred_at_unix_ms"] if existing is not None
                       else int(time.time()*1000))
        payload = {"schema_version": "1", "outcome": outcome,
            "tenant_id": row["tenant"], "request_id": context["request_id"],
            "key_id": context["key_id"],
            "execution_attempt_id": row["attempt"], "reservation_id": row["reservation"],
            "usage_event_id": row["usage_event"], "request_sha256": context["request_sha256"],
            "policy_revision": context["policy_revision"],
            "started_at_unix_ms": row["started_at_ms"], "occurred_at_unix_ms": occurred_at,
            "runtime": runtime, "observation": observation}
        encoded = _json(payload)
        if existing is not None:
            if existing["payload_json"] != encoded:
                raise JournalConflict("terminal evidence cannot be replaced")
            return Receipt(existing["receipt_id"], encoded, existing["payload_sha256"])
        receipt = Receipt(str(uuid.uuid4()), encoded, hashlib.sha256(encoded.encode()).hexdigest())
        db.execute("INSERT INTO receipts(receipt_id,tenant,attempt,payload_json,payload_sha256) VALUES(?,?,?,?,?)",
                   (receipt.receipt_id, row["tenant"], row["attempt"], encoded, receipt.payload_sha256))
        db.execute("UPDATE executions SET state=?,updated_ms=? WHERE tenant=? AND attempt=?",
                   (outcome, int(time.time()*1000), row["tenant"], row["attempt"]))
        return receipt

    def completed(self, context, usage):
        usage = dict(usage)
        expected = {"requests", "question_rows", "encoded_tokens", "output_tokens", "queue_ms", "execution_ms"}
        if set(usage) != expected:
            raise ValueError("completed receipt requires observed host usage; no inferred accelerator time")
        for name, value in usage.items():
            self._integer(value, name)
        with self._transaction() as db:
            row = self._row(db, context)
            if row["state"] not in ("running", "completed"):
                raise JournalConflict("completion requires a recorded execution start")
            if (usage["requests"] != 1 or usage["output_tokens"] != 0
                    or usage["question_rows"] != row["question_rows"]
                    or usage["encoded_tokens"] != row["encoded_tokens"]
                    or usage["queue_ms"] != row["queue_ms"]):
                raise JournalConflict("observed usage differs from recorded execution")
            return self._terminal(db, row, "completed", usage)

    def failed(self, context, execution_ms):
        self._integer(execution_ms, "execution_ms")
        with self._transaction() as db:
            row = self._row(db, context)
            if row["state"] not in ("running", "failed"):
                raise JournalConflict("failure evidence requires a recorded execution start")
            return self._terminal(db, row, "failed", {"queue_ms": row["queue_ms"], "execution_ms": execution_ms})

    def not_started(self, context, reason):
        if reason not in ("queue_rejected", "cancelled", "expired", "runtime_failed"):
            raise ValueError("unsupported nonexecution reason")
        with self._transaction() as db:
            row = self._row(db, context)
            if row["state"] != "admitted":
                return False  # Cancellation cannot erase an already started execution.
            self._terminal(db, row, "not_started", {"reason": reason})
        return True

    def delivery(self, context, outcome):
        if outcome not in ("cancelled", "expired", "failed", "result_ready"):
            raise ValueError("unsupported local delivery outcome")
        with self._transaction() as db:
            row = self._row(db, context)
            if row["delivery"] not in ("pending", outcome):
                raise JournalConflict("local delivery outcome cannot be replaced")
            db.execute("UPDATE executions SET delivery=?,updated_ms=? WHERE tenant=? AND attempt=?",
                       (outcome, int(time.time()*1000), row["tenant"], row["attempt"]))

    def pending_receipts(self, limit=100):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("receipt batch limit must be between 1 and 1000")
        with self._connect() as db:
            rows = db.execute("SELECT receipt_id,payload_json,payload_sha256 FROM receipts WHERE acknowledged=0 ORDER BY rowid LIMIT ?", (limit,)).fetchall()
        if any(hashlib.sha256(row["payload_json"].encode()).hexdigest() != row["payload_sha256"] for row in rows):
            raise JournalConflict("persisted receipt digest does not match payload")
        return [Receipt(*row) for row in rows]

    def acknowledge(self, receipt_id, payload_sha256):
        """Apply a platform-authenticated acknowledgment of these exact bytes.

        The caller authenticates the platform response; receipt ID alone is never
        sufficient. Repeated identical acknowledgments are safe.
        """
        with self._transaction() as db:
            row = db.execute("SELECT * FROM receipts WHERE receipt_id=?", (receipt_id,)).fetchone()
            if (row is None or row["payload_sha256"] != payload_sha256
                    or hashlib.sha256(row["payload_json"].encode()).hexdigest() != payload_sha256):
                raise JournalConflict("receipt acknowledgment does not match persisted payload")
            db.execute("UPDATE receipts SET acknowledged=1 WHERE receipt_id=?", (receipt_id,))

    def check_startup(self):
        """Refuse model startup while earlier execution needs reconciliation.

        This read-only guard is not a device lease or a cross-process startup
        lock. The supervisor still fences the previous owner before launching.
        """
        with self._connect() as db:
            pending = db.execute(
                "SELECT 1 FROM executions WHERE state IN ('admitted','running') LIMIT 1"
            ).fetchone()
        if pending is not None:
            raise JournalConflict("execution recovery is required before startup")

    def unresolved(self):
        """Return crash-reconciliation candidates; never label them completed usage."""
        with self._connect() as db:
            rows = db.execute("SELECT tenant,attempt,reservation,usage_event,state,delivery FROM executions WHERE state IN ('admitted','running') ORDER BY rowid").fetchall()
        return [dict(row) for row in rows]

    def status(self, context):
        with self._connect() as db:
            row = self._row(db, context)
            return {key: row[key] for key in ("state", "delivery", "question_rows", "encoded_tokens")}
