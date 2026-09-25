"""Wire fixtures are produced by the real journal under a synthetic fixed clock."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator
from laya_tt.contracts import load_schema
from laya_tt.ledger import ExecutionJournal, JournalConflict


FIXTURES = Path(__file__).resolve().parents[1] / "schemas/fixtures/execution-receipts"
CONTEXT = dict(schema_version="1", request_id="request", execution_attempt_id="attempt",
    usage_event_id="usage", tenant_id="tenant", key_id="key", runtime_generation="generation",
    policy_revision="policy", reservation_id="reservation", model="laya-english",
    checkpoint_revision="a" * 40, issued_at_unix_ms=1000, deadline_unix_ms=3000,
    max_question_rows=2, max_encoded_tokens=20, request_sha256="b" * 64)
RUNTIME = dict(model="laya-english", checkpoint_revision="a" * 40,
    runtime_revision="fixture-build", runtime_generation="generation", backend="cpu-reference")
USAGE = dict(requests=1, question_rows=2, encoded_tokens=17, output_tokens=0,
    queue_ms=3, execution_ms=7)


def produce(journal, outcome):
    with patch("laya_tt.ledger.time.time", return_value=2):
        journal.admitted(CONTEXT, 2, 17, RUNTIME)
        if outcome == "not_started":
            journal.not_started(CONTEXT, "cancelled")
        else:
            journal.started(CONTEXT, 3)
            if outcome == "completed":
                journal.completed(CONTEXT, USAGE)
            else:
                journal.failed(CONTEXT, 7)
    return journal.pending_receipts()[0]


class ExecutionReceiptTests(unittest.TestCase):
    def test_real_journal_output_matches_portable_fixtures_and_survives_reopen(self):
        schema = load_schema("execution-receipt")
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        with tempfile.TemporaryDirectory() as root:
            for outcome in ("completed", "failed", "not_started"):
                with self.subTest(outcome=outcome):
                    path = Path(root) / (outcome + ".sqlite")
                    receipt = produce(ExecutionJournal(path), outcome)
                    self.assertEqual(receipt.payload_json,
                        (FIXTURES / (outcome + ".json")).read_text(encoding="utf-8").rstrip("\n"))
                    validator.validate(json.loads(receipt.payload_json))
                    reopened = ExecutionJournal(path)
                    self.assertEqual(reopened.pending_receipts(), [receipt])
                    self.assertEqual(receipt.payload_sha256,
                        hashlib.sha256(receipt.payload_json.encode()).hexdigest())
                    reopened.acknowledge(receipt.receipt_id, receipt.payload_sha256)
                    self.assertEqual(reopened.pending_receipts(), [])

    def test_outcomes_cannot_invent_usage_or_omit_start_evidence(self):
        validator = Draft202012Validator(load_schema("execution-receipt"))
        completed = json.loads((FIXTURES / "completed.json").read_text(encoding="utf-8"))
        mutations = [
            {"outcome": "failed"}, {"outcome": "not_started"},
            {"started_at_unix_ms": None}, {"tenant_id": ""},
            {"usage_event_id": "x" * 129}, {"request_sha256": "z" * 64},
            {"occurred_at_unix_ms": 2 ** 64}, {"prompt": "private input"},
            {"runtime": dict(RUNTIME, backend="unverified")},
            {"observation": dict(USAGE, output_tokens=1)},
            {"observation": dict(USAGE, accelerator_ms=7)},
            {"observation": dict(USAGE, encoded_tokens=32769)},
        ]
        for change in mutations:
            with self.subTest(fields=list(change)):
                candidate = copy.deepcopy(completed)
                candidate.update(change)
                self.assertFalse(validator.is_valid(candidate))
        not_started = json.loads((FIXTURES / "not_started.json").read_text(encoding="utf-8"))
        not_started["started_at_unix_ms"] = 2000
        self.assertFalse(validator.is_valid(not_started))

    def test_invalid_receipt_rolls_back_terminal_state_and_outbox(self):
        with tempfile.TemporaryDirectory() as root:
            journal = ExecutionJournal(Path(root) / "journal.sqlite")
            journal.admitted(CONTEXT, 2, 17, RUNTIME)
            journal.started(CONTEXT, 3)
            with self.assertRaisesRegex(JournalConflict, "wire contract"):
                journal.completed(CONTEXT, dict(USAGE, execution_ms=2 ** 64))
            self.assertEqual(journal.status(CONTEXT)["state"], "running")
            self.assertEqual(journal.pending_receipts(), [])
            self.assertEqual(len(journal.unresolved()), 1)
