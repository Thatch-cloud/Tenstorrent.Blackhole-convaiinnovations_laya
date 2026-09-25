from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import laya_tt.ledger as ledger_module
from laya_tt.admission import AdmissionAdapter, PreparedInput
from laya_tt.asgi import DecisionASGI
from laya_tt.ledger import ExecutionJournal, JournalConflict, JournalReplay
from laya_tt.service import DecisionService, RuntimeIdentity
from laya_tt.worker import QueueFull


def claims(tenant="tenant", attempt="attempt"):
    now = int(time.time()*1000)
    return dict(schema_version="1", request_id="request", execution_attempt_id=attempt,
        usage_event_id="usage", tenant_id=tenant, key_id="key", runtime_generation="generation",
        policy_revision="policy", reservation_id="reservation", model="laya-english",
        checkpoint_revision="a"*40, issued_at_unix_ms=now-1000, deadline_unix_ms=now+60000,
        max_question_rows=1, max_encoded_tokens=10, request_sha256="b"*64)


RUNTIME = RuntimeIdentity("a"*40, "build", "generation", "cpu-reference")
USAGE = dict(requests=1, question_rows=1, encoded_tokens=5, output_tokens=0, queue_ms=3, execution_ms=7)


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "execution.sqlite"
        self.context = claims()
        self.journal = ExecutionJournal(self.path)

    def admit(self):
        self.journal.admitted(self.context, 1, 5, asdict(RUNTIME))

    def test_restart_replay_and_tenant_scoped_identity(self):
        self.admit()
        reopened = ExecutionJournal(self.path)
        with self.assertRaises(JournalReplay):
            reopened.admitted(self.context, 1, 5, asdict(RUNTIME))
        for changes in ({"reservation_id": "different", "usage_event_id": "different"},
                        {"execution_attempt_id": "different", "usage_event_id": "different"},
                        {"execution_attempt_id": "different", "reservation_id": "different"}):
            with self.assertRaises(JournalReplay):
                reopened.admitted(dict(self.context, **changes), 1, 5, asdict(RUNTIME))
        reopened.admitted(dict(self.context, tenant_id="other"), 1, 5, asdict(RUNTIME))
        self.assertEqual(len(reopened.unresolved()), 2)

    def test_start_is_once_and_exact_context_bound(self):
        self.admit()
        with self.assertRaises(JournalConflict):
            self.journal.started(dict(self.context, request_sha256="c"*64), 3)
        self.assertTrue(self.journal.started(self.context, 3))
        self.assertFalse(self.journal.started(self.context, 3))
        self.assertFalse(self.journal.not_started(self.context, "cancelled"))
        self.assertEqual(self.journal.status(self.context)["state"], "running")

    def test_atomic_completion_idempotence_and_payload_bound_ack(self):
        self.admit()
        self.journal.started(self.context, 3)
        receipt = self.journal.completed(self.context, USAGE)
        reopened = ExecutionJournal(self.path)
        with patch("laya_tt.ledger.time.time", return_value=time.time()+3600):
            self.assertEqual(reopened.completed(self.context, USAGE), receipt)
        self.assertEqual(reopened.unresolved(), [])
        with self.assertRaises(JournalConflict):
            reopened.completed(self.context, dict(USAGE, execution_ms=8))
        with self.assertRaises(JournalConflict):
            reopened.acknowledge(receipt.receipt_id, "0"*64)
        self.assertEqual(reopened.pending_receipts(), [receipt])
        reopened.acknowledge(receipt.receipt_id, receipt.payload_sha256)
        reopened.acknowledge(receipt.receipt_id, receipt.payload_sha256)
        self.assertEqual(ExecutionJournal(self.path).pending_receipts(), [])
        with self.assertRaises(JournalReplay):
            reopened.admitted(self.context, 1, 5, asdict(RUNTIME))

    def test_unresolved_and_failed_records_are_not_completed_usage(self):
        self.admit()
        self.assertEqual(self.journal.pending_receipts(), [])
        self.assertEqual(self.journal.unresolved()[0]["state"], "admitted")
        self.journal.started(self.context, 3)
        self.assertEqual(self.journal.unresolved()[0]["state"], "running")
        receipt = self.journal.failed(self.context, 7)
        payload = json.loads(receipt.payload_json)
        self.assertEqual(payload["outcome"], "failed")
        self.assertNotIn("encoded_tokens", payload["observation"])
        self.assertNotIn("accelerator_ms", payload["observation"])
        self.assertEqual(self.journal.unresolved(), [])

    def test_nonexecution_receipt_prevents_start_without_claiming_usage(self):
        self.admit()
        self.assertTrue(self.journal.not_started(self.context, "cancelled"))
        self.assertFalse(self.journal.started(self.context, 0))
        payload = json.loads(self.journal.pending_receipts()[0].payload_json)
        self.assertEqual(payload["observation"], {"reason": "cancelled"})
        self.assertEqual(payload["outcome"], "not_started")

    def test_usage_mismatch_or_invented_accelerator_observation_rejected(self):
        self.admit()
        self.journal.started(self.context, 3)
        with self.assertRaises(JournalConflict):
            self.journal.completed(self.context, dict(USAGE, encoded_tokens=6))
        with self.assertRaises(ValueError):
            self.journal.completed(self.context, dict(USAGE, accelerator_ms=7))
        self.assertEqual(self.journal.pending_receipts(), [])

    def test_corrupt_outbox_payload_cannot_be_sent_or_acknowledged(self):
        self.admit()
        self.journal.started(self.context, 3)
        receipt = self.journal.completed(self.context, USAGE)
        db = sqlite3.connect(self.path)
        try:
            db.execute("UPDATE receipts SET payload_json='{}'")
            db.commit()
        finally:
            db.close()
        with self.assertRaises(JournalConflict):
            self.journal.pending_receipts()
        with self.assertRaises(JournalConflict):
            self.journal.acknowledge(receipt.receipt_id, receipt.payload_sha256)

    def test_actual_process_crashes_reopen_with_only_committed_evidence(self):
        # Terminate a real child without cleanup at three lifecycle boundaries.
        script = '''
import dataclasses, importlib.util, json, os, sys
sys.path.insert(0, sys.argv[1])
spec=importlib.util.spec_from_file_location("laya_tt.ledger",sys.argv[2])
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
j=module.ExecutionJournal(sys.argv[3]); c=json.loads(sys.argv[4]); r=json.loads(sys.argv[5]); stage=sys.argv[6]
j.admitted(c,1,5,r)
if stage in ("running","completed"): j.started(c,3)
if stage == "completed": j.completed(c,json.loads(sys.argv[7]))
os._exit(73)
'''
        import laya_tt
        src = str(Path(laya_tt.__file__).parent.parent)
        for stage in ("admitted", "running", "completed"):
            path = Path(self.temp.name) / (stage + ".sqlite")
            run = subprocess.run([sys.executable, "-B", "-c", script, src, ledger_module.__file__,
                str(path), json.dumps(self.context), json.dumps(asdict(RUNTIME)), stage, json.dumps(USAGE)],
                capture_output=True, timeout=15)
            self.assertEqual(run.returncode, 73, run.stderr.decode())
            reopened = ExecutionJournal(path)
            with self.assertRaises(JournalReplay):
                reopened.admitted(self.context, 1, 5, asdict(RUNTIME))
            if stage == "completed":
                self.assertEqual(reopened.unresolved(), [])
                self.assertEqual(json.loads(reopened.pending_receipts()[0].payload_json)["outcome"], "completed")
            else:
                self.assertEqual(reopened.unresolved()[0]["state"], stage)
                self.assertEqual(reopened.pending_receipts(), [])


class JournalServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "journal.sqlite"
        self.context = claims()
        self.raw = json.dumps({"model":"laya-english","state":"private-input",
            "questions":{"q":{"type":"noul","instructions":"private-question"}}}).encode()
        self.context["request_sha256"] = hashlib.sha256(self.raw).hexdigest()

    def service(self, journal, backend):
        admission = AdmissionAdapter(verify_grant=lambda grant: self.context,
            prepare=lambda request: PreparedInput(1,5,request["state"]),
            consume_reservation=lambda *args: True,  # Explicit test authority fixture.
            checkpoint_revision=RUNTIME.checkpoint_revision,
            runtime_generation=RUNTIME.runtime_generation, policy_revision="policy")
        service = DecisionService(admission=admission,backend=backend,runtime=RUNTIME,receipt_sink=journal)
        service.start(lambda: True)
        self.addCleanup(lambda: service.drain(timeout=3))
        return service

    @staticmethod
    def output(payload):
        return {"answers":{"q":{"type":"noul","noul":.7,"confidence":.7,
            "answer_confidence":.7,"action":{"act_probability":.5}}},"encoded_tokens":5}

    def test_cancelled_inflight_execution_still_has_durable_observed_receipt(self):
        journal = ExecutionJournal(self.path)
        entered, release = threading.Event(), threading.Event()
        def backend(payload):
            entered.set();release.wait(3)
            return self.output(payload)
        service = self.service(journal, backend)
        self.addCleanup(release.set)
        future = service.submit(self.raw,b"verified-test-grant")
        self.assertTrue(entered.wait(1))
        future.cancel()
        self.assertEqual(journal.status(self.context)["state"],"running")
        release.set()
        self.assertTrue(service.drain(timeout=3))
        reopened = ExecutionJournal(self.path)
        self.assertEqual(reopened.status(self.context),dict(state="completed",delivery="cancelled",question_rows=1,encoded_tokens=5))
        payload = reopened.pending_receipts()[0].payload_json
        self.assertNotIn("private-input",payload)
        self.assertNotIn("private-question",payload)
        self.assertEqual(json.loads(payload)["observation"]["encoded_tokens"],5)

    def test_completion_transaction_failure_withholds_response_and_rolls_back_outbox(self):
        class BrokenCommit(ExecutionJournal):
            def _terminal(self, db, row, outcome, observation):
                result = super()._terminal(db,row,outcome,observation)
                if outcome == "completed":
                    raise OSError("simulated disk failure before commit")
                return result
        journal=BrokenCommit(self.path)
        executed=[]
        def backend(payload):
            executed.append(True)
            return self.output(payload)
        service=self.service(journal,backend)
        with self.assertRaises(OSError):
            service.submit(self.raw,b"verified-test-grant").result(2)
        self.assertEqual(executed,[True])
        self.assertFalse(service.readiness()["ready"])
        service.drain(timeout=2)
        reopened=ExecutionJournal(self.path)
        self.assertEqual(reopened.pending_receipts(),[])
        self.assertEqual(reopened.unresolved()[0]["state"],"running")

    def test_cancel_wins_atomic_start_transition_and_backend_never_runs(self):
        entered, release = threading.Event(), threading.Event()
        class GatedStartJournal(ExecutionJournal):
            def started(self, context, queue_ms):
                entered.set()
                release.wait(3)
                return super().started(context, queue_ms)
        journal=GatedStartJournal(self.path)
        executed=[]
        def backend(payload):
            executed.append(True)
            return self.output(payload)
        service=self.service(journal,backend)
        self.addCleanup(release.set)
        future=service.submit(self.raw,b"verified-test-grant")
        self.assertTrue(entered.wait(1))
        self.assertTrue(future.cancel())
        self.assertEqual(journal.status(self.context)["state"],"not_started")
        release.set()
        self.assertTrue(service.drain(timeout=3))
        self.assertEqual(executed,[])
        payload=json.loads(journal.pending_receipts()[0].payload_json)
        self.assertEqual(payload["observation"],{"reason":"cancelled"})

    def test_queue_rejection_records_nonexecution_after_authority_consumption(self):
        journal=ExecutionJournal(self.path)
        service=self.service(journal,self.output)
        def full(*args,**kwargs):
            raise QueueFull("test queue full")
        service._worker.submit=full
        with self.assertRaises(QueueFull):
            service.submit(self.raw,b"verified-test-grant")
        receipt=json.loads(journal.pending_receipts()[0].payload_json)
        self.assertEqual(receipt["outcome"],"not_started")
        self.assertEqual(receipt["observation"],{"reason":"queue_rejected"})
        self.assertTrue(service.readiness()["ready"])

    def test_replay_rejected_without_withdrawing_healthy_runtime(self):
        journal=ExecutionJournal(self.path)
        executed=[]
        def backend(payload):
            executed.append(True)
            return self.output(payload)
        service=self.service(journal,backend)
        service.submit(self.raw,b"verified-test-grant").result(2)
        with self.assertRaises(JournalReplay) as raised:
            service.submit(self.raw,b"verified-test-grant")
        self.assertEqual(DecisionASGI._failure(raised.exception),(409,"duplicate_request"))
        self.assertTrue(service.readiness()["ready"])
        self.assertEqual(executed,[True])


if __name__ == "__main__":
    unittest.main()
