import hashlib
import json
import os
from pathlib import Path
import threading
import time
import unittest

from jsonschema import Draft202012Validator

from laya_tt.admission import AdmissionAdapter, AdmissionRejected, PreparedInput
from laya_tt.contracts import load_schema
from laya_tt.service import (
    AdmissionCapacityExceeded, DecisionService, InvalidBackendResult,
    RuntimeIdentity, ServiceNotReady,
)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.grants = {}
        self.consumed = set()
        self.verifications = 0
        self.entered, self.release = threading.Event(), threading.Event()
        self.calls = []

    def envelope(self, tenant, state, attempt="same-attempt", **overrides):
        raw = json.dumps({"model": "laya-english", "state": state,
            "questions": {"same-question": {"type": "noul", "instructions": "Evaluate"}}}).encode()
        envelope = f"authenticated:{tenant}:{attempt}".encode()
        now = int(time.time() * 1000)
        self.grants[envelope] = dict(schema_version="1", request_id="same-request",
            execution_attempt_id=attempt, usage_event_id=f"usage-{tenant}-{attempt}",
            tenant_id=tenant, key_id="key", runtime_generation="generation",
            policy_revision="policy", reservation_id=f"reservation-{tenant}-{attempt}",
            model="laya-english", checkpoint_revision="a" * 40,
            issued_at_unix_ms=now - 100, deadline_unix_ms=now + 5000,
            max_question_rows=1, max_encoded_tokens=20,
            request_sha256=hashlib.sha256(raw).hexdigest())
        self.grants[envelope].update(overrides)
        return raw, envelope

    def verify(self, envelope):
        self.verifications += 1
        if envelope not in self.grants:
            raise PermissionError("unauthenticated envelope")
        return self.grants[envelope]

    def consume(self, context, rows, tokens):
        self.assertEqual((rows, tokens), (1, 5))
        key = context["reservation_id"]
        if key in self.consumed:
            return False
        self.consumed.add(key)
        return True

    def backend(self, state):
        self.calls.append(state)
        if state == "hold":
            self.entered.set()
            if not self.release.wait(3):
                raise RuntimeError("test hold timeout")
        value = 1. if state == "first" else 0.
        return {"answers": {"same-question": {"type": "noul", "noul": value,
            "confidence": 1., "answer_confidence": 1., "action": {"act_probability": .5}}},
            "encoded_tokens": 5}

    def service(self, *, backend=None, prepare=None, ready=True, **kwargs):
        admission = AdmissionAdapter(verify_grant=self.verify,
            prepare=prepare or (lambda request: PreparedInput(1, 5, request["state"])),
            consume_reservation=self.consume, checkpoint_revision="a" * 40,
            runtime_generation="generation", policy_revision="policy")
        service = DecisionService(admission=admission, backend=backend or self.backend,
            runtime=RuntimeIdentity("a" * 40, "runtime-build", "generation", "cpu-reference"),
            **kwargs)
        self.addCleanup(lambda: service.drain(timeout=3, cancel_queued=True))
        self.addCleanup(self.release.set)
        if ready:
            service.start(lambda: True)  # Deterministic fake model fixture only.
        return service

    def test_interleaved_tenants_same_ids_keep_answers_and_observed_usage(self):
        service = self.service()
        hold = service.submit(*self.envelope("blocker", "hold"))
        self.assertTrue(self.entered.wait(1))
        first = service.submit(*self.envelope("tenant-a", "first"))
        second = service.submit(*self.envelope("tenant-b", "second"))
        release_after = time.monotonic() + .03
        while time.monotonic() < release_after:
            time.sleep(.005)
        self.release.set()
        hold.result(2)
        a, b = first.result(2), second.result(2)
        validator = Draft202012Validator(load_schema("decision-response"))
        for response in (a, b):
            validator.validate(response)
            self.assertEqual(response["request_id"], "same-request")
            self.assertEqual(list(response["answers"]), ["same-question"])
            self.assertEqual(response["backend"], "cpu-reference")
            self.assertEqual(response["usage"]["encoded_tokens"], 5)
            self.assertGreaterEqual(response["usage"]["queue_ms"], 20)
            self.assertNotIn("accelerator_ms", response["usage"])
            self.assertNotIn("tenant_id", response)
        self.assertEqual(a["answers"]["same-question"]["noul"], 1.)
        self.assertEqual(b["answers"]["same-question"]["noul"], 0.)
        a["answers"]["same-question"]["noul"] = .4
        self.assertEqual(b["answers"]["same-question"]["noul"], 0.)
        self.assertEqual(self.calls, ["hold", "first", "second"])

    def test_startup_gate_and_failed_self_test(self):
        service = self.service(ready=False)
        with self.assertRaises(ServiceNotReady):
            service.submit(*self.envelope("a", "first"))
        self.assertEqual(self.verifications, 0)
        self.assertEqual(service.readiness()["state"], "starting")
        with self.assertRaises(ServiceNotReady):
            service.start(lambda: False)
        self.assertEqual(service.readiness()["state"], "failed")
        with self.assertRaises(ServiceNotReady):
            service.start(lambda: True)

    def test_no_authentication_fallback(self):
        service = self.service()
        raw, _ = self.envelope("a", "first")
        with self.assertRaises(AdmissionRejected):
            service.submit(raw, b"forged")
        self.assertEqual(self.calls, [])

    def test_inflight_cancel_and_drain_do_not_release_backend_early(self):
        service = self.service()
        active = service.submit(*self.envelope("a", "hold"))
        self.assertTrue(self.entered.wait(1))
        queued = service.submit(*self.envelope("b", "second"))
        self.assertTrue(active.cancel())
        self.assertFalse(service.drain(timeout=.01, cancel_queued=True))
        self.assertTrue(queued.cancelled())
        self.assertEqual(service.readiness()["state"], "draining")
        with self.assertRaises(ServiceNotReady):
            service.submit(*self.envelope("c", "first"))
        self.release.set()
        self.assertTrue(service.drain(timeout=2))
        self.assertEqual(service.readiness()["state"], "stopped")
        self.assertEqual(self.calls, ["hold"])

    def test_expired_request_and_replay_do_not_execute(self):
        service = self.service()
        raw, envelope = self.envelope("a", "first", deadline_unix_ms=int(time.time()*1000)-1)
        with self.assertRaises(AdmissionRejected):
            service.submit(raw, envelope)
        raw, envelope = self.envelope("b", "second")
        service.submit(raw, envelope).result(1)
        with self.assertRaises(AdmissionRejected):
            service.submit(raw, envelope)
        self.assertEqual(self.calls, ["second"])

    def test_bad_backend_token_accounting_withdraws_readiness(self):
        def bad(state):
            result = self.backend(state)
            result["encoded_tokens"] = 6
            return result
        service = self.service(backend=bad)
        with self.assertRaises(InvalidBackendResult):
            service.submit(*self.envelope("a", "first")).result(1)
        self.assertEqual(service.readiness()["state"], "failed")
        with self.assertRaises(ServiceNotReady):
            service.submit(*self.envelope("b", "first"))

    def test_nonfinite_and_changed_question_identity_rejected(self):
        for mode in ("nan", "identity"):
            def bad(state, mode=mode):
                result = self.backend(state)
                if mode == "nan":
                    result["answers"]["same-question"]["noul"] = float("nan")
                else:
                    result["answers"]["other-question"] = result["answers"].pop("same-question")
                return result
            service = self.service(backend=bad)
            with self.subTest(mode=mode), self.assertRaises(InvalidBackendResult):
                service.submit(*self.envelope(mode, "first")).result(1)

    def test_drain_waits_for_admission_and_closes_racing_submission(self):
        preparing, prepared = threading.Event(), threading.Event()
        def prepare(request):
            preparing.set()
            prepared.wait(2)
            return PreparedInput(1, 5, request["state"])
        service = self.service(prepare=prepare)
        self.addCleanup(prepared.set)
        failures = []
        def submit():
            try:
                service.submit(*self.envelope("a", "first"))
            except Exception as exc:
                failures.append(exc)
        thread = threading.Thread(target=submit)
        thread.start()
        self.assertTrue(preparing.wait(1))
        self.assertFalse(service.drain(timeout=.01))
        prepared.set()
        thread.join(2)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], ServiceNotReady)
        self.assertTrue(service.drain(timeout=2))
        self.assertEqual(self.calls, [])

    def test_waiting_for_worker_does_not_hold_readiness_lock(self):
        service = self.service()
        entered_submit, read_ready = threading.Event(), threading.Event()
        original_submit = service._worker.submit
        results = []
        def observed_submit(*args, **kwargs):
            entered_submit.set()
            return original_submit(*args, **kwargs)
        service._worker.submit = observed_submit
        raw, envelope = self.envelope("a", "first")
        submitter = threading.Thread(target=lambda: results.append(service.submit(raw, envelope)))
        reader = threading.Thread(target=lambda: (service.readiness(), read_ready.set()))
        with service._worker._cv:
            submitter.start()
            self.assertTrue(entered_submit.wait(1))
            reader.start()
            responsive = read_ready.wait(1)
        submitter.join(2)
        reader.join(2)
        self.assertTrue(responsive, "worker contention must not block readiness callbacks")
        self.assertEqual(results[0].result(2)["backend"], "cpu-reference")

    def test_bounded_preparation_rejects_before_verification_or_reservation(self):
        preparing, release_prepare = threading.Event(), threading.Event()
        def prepare(request):
            preparing.set()
            if not release_prepare.wait(2):
                raise RuntimeError("test preparation timeout")
            return PreparedInput(1, 5, request["state"])
        service = self.service(prepare=prepare, max_admitting=1)
        self.addCleanup(release_prepare.set)
        raw, envelope = self.envelope("a", "first")
        results = []
        submitter = threading.Thread(target=lambda: results.append(service.submit(raw, envelope)))
        submitter.start()
        self.assertTrue(preparing.wait(1))
        with self.assertRaises(AdmissionCapacityExceeded):
            service.submit(*self.envelope("b", "second"))
        self.assertEqual(self.verifications, 1)
        self.assertEqual(self.consumed, set())
        release_prepare.set()
        submitter.join(2)
        results[0].result(2)
        service.submit(*self.envelope("b", "second")).result(2)
        self.assertEqual(self.calls, ["first", "second"])

    def test_admission_failure_releases_preparation_slot(self):
        service = self.service(max_admitting=1)
        raw, envelope = self.envelope("a", "first")
        with self.assertRaises(AdmissionRejected):
            service.submit(raw, b"forged")
        self.assertEqual(self.consumed, set())
        self.assertEqual(service.submit(raw, envelope).result(2)["backend"], "cpu-reference")

    def test_max_admitting_must_be_positive_integer(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.service(max_admitting=value)


@unittest.skipUnless(os.environ.get("LAYA_CPU_INTEGRATION") == "1",
                     "optional service integration requires pinned CPU checkpoint")
class RealCpuServiceTests(unittest.TestCase):
    def test_two_tenants_actual_model_same_question_and_attempt_ids(self):
        from laya_tt.cpu_backend import load_cpu_backend
        root = Path(__file__).resolve().parents[1]
        manifest_path = root / "configs/checkpoint-lock.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        backend = load_cpu_backend(manifest_path, root=root)
        checkpoint = manifest["checkpoint"]["revision"]
        grants, consumed = {}, set()
        cases = []
        for tenant, state in (("one", "The sky is blue."), ("two", "The sky is red.")):
            request = {"model": "laya-english", "state": state, "questions": {
                "same-question": {"type": "choice", "instructions": "What color is the sky?",
                                  "criteria": ["blue", "red"]}}}
            raw, envelope = json.dumps(request).encode(), tenant.encode()
            now = int(time.time() * 1000)
            grants[envelope] = dict(schema_version="1", request_id="same-request",
                execution_attempt_id="same-attempt", usage_event_id=f"usage-{tenant}",
                tenant_id=tenant, key_id="test-key", runtime_generation="cpu-generation",
                policy_revision="test-policy", reservation_id=f"reservation-{tenant}",
                model="laya-english", checkpoint_revision=checkpoint,
                issued_at_unix_ms=now-100, deadline_unix_ms=now+120000,
                max_question_rows=64, max_encoded_tokens=32768,
                request_sha256=hashlib.sha256(raw).hexdigest())
            prepared = backend.prepare(request)
            expected = backend.execute(prepared.payload)
            cases.append((raw, envelope, expected))
        def consume(context, rows, tokens):
            key = context["reservation_id"]
            if key in consumed:
                return False
            consumed.add(key)
            return True
        admission = AdmissionAdapter(verify_grant=lambda envelope: grants[envelope],
            prepare=backend.prepare, consume_reservation=consume,
            checkpoint_revision=checkpoint, runtime_generation="cpu-generation",
            policy_revision="test-policy")
        service = DecisionService(admission=admission, backend=backend,
            runtime=RuntimeIdentity(checkpoint, "cpu-integration-test", "cpu-generation", "cpu-reference"))
        self.addCleanup(lambda: service.drain(timeout=120))
        def self_test():
            request = json.loads(cases[0][0])
            return backend.execute(backend.prepare(request).payload)["answers"] == cases[0][2]["answers"]
        service.start(self_test)
        futures = [service.submit(raw, envelope) for raw, envelope, _ in cases]
        for future, (_, _, expected) in zip(futures, cases):
            response = future.result(120)
            self.assertEqual(response["answers"], expected["answers"])
            self.assertEqual(response["usage"]["encoded_tokens"], expected["encoded_tokens"])
            self.assertEqual(response["backend"], "cpu-reference")
            self.assertNotIn("accelerator_ms", response["usage"])


if __name__ == "__main__":
    unittest.main()
