import copy
import hashlib
import json
from pathlib import Path
import unittest

from laya_tt.admission import AdmissionAdapter, AdmissionRejected, PreparedInput


SCHEMAS = Path(__file__).resolve().parents[1] / "src/laya_tt/schemas"


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.request = {"model": "laya-english", "state": {"tenant_id": "untrusted-state"},
                        "questions": {"q": {"type": "noul", "instructions": "Check this"}}}
        self.raw = json.dumps(self.request).encode()
        self.claims = {
            "schema_version": "1", "request_id": "request", "execution_attempt_id": "attempt",
            "usage_event_id": "usage", "tenant_id": "verified-tenant", "key_id": "key",
            "runtime_generation": "generation", "policy_revision": "policy",
            "reservation_id": "reservation", "model": "laya-english",
            "checkpoint_revision": "a" * 40, "issued_at_unix_ms": 9000,
            "deadline_unix_ms": 11000, "max_question_rows": 4,
            "max_encoded_tokens": 100, "request_sha256": hashlib.sha256(self.raw).hexdigest(),
        }
        self.verified = []
        self.consumed = []
        self.prepared = []
        self.now = 10.
        self.mono = 50.

    def verify(self, envelope):
        self.verified.append(envelope)
        if envelope != b"platform-authenticated-envelope":
            raise PermissionError("private verification details must not escape")
        return self.claims

    def prepare(self, request):
        self.prepared.append(request)
        return PreparedInput(1, 37, {"tokens": [1, 2, 3]})

    def consume(self, context, rows, tokens):
        self.consumed.append((context, rows, tokens))
        return len(self.consumed) == 1

    def adapter(self, **kwargs):
        config = dict(schema_dir=SCHEMAS, verify_grant=self.verify, prepare=self.prepare,
                      consume_reservation=self.consume, checkpoint_revision="a" * 40,
                      runtime_generation="generation", policy_revision="policy",
                      wall_time=lambda: self.now, monotonic=lambda: self.mono)
        config.update(kwargs)
        return AdmissionAdapter(**config)

    def admit(self, **kwargs):
        return self.adapter(**kwargs).admit(self.raw, b"platform-authenticated-envelope")

    def test_verified_context_owned_and_immutable_with_actual_cost(self):
        admitted = self.admit()
        self.assertEqual(admitted.context["tenant_id"], "verified-tenant")
        self.assertEqual(admitted.request["state"]["tenant_id"], "untrusted-state")
        self.assertEqual(admitted.deadline, 51.)
        self.assertEqual(self.consumed[0][1:], (1, 37))
        self.assertEqual(admitted.prepared.payload["tokens"], (1, 2, 3))
        self.claims["tenant_id"] = "later-mutation"
        self.assertEqual(admitted.context["tenant_id"], "verified-tenant")
        with self.assertRaises(TypeError):
            admitted.context["tenant_id"] = "bad"
        with self.assertRaises(TypeError):
            admitted.request["state"]["tenant_id"] = "bad"

    def test_submit_uses_verified_tenant_attempt_and_actual_tokens(self):
        class SpyWorker:
            def submit(self, *args, **kwargs):
                self.args, self.kwargs = args, kwargs
                return "future"
        worker = SpyWorker()
        result = self.adapter().submit(worker, self.raw, b"platform-authenticated-envelope")
        self.assertEqual(result, "future")
        self.assertEqual(worker.args[:2], ("verified-tenant", "attempt"))
        self.assertEqual(worker.kwargs, {"cost": 37, "deadline": 51.})

    def test_unverified_cross_tenant_claim_does_not_reach_preparation(self):
        forged = json.dumps(dict(self.claims, tenant_id="victim")).encode()
        with self.assertRaisesRegex(AdmissionRejected, "grant verification failed"):
            self.adapter().admit(self.raw, forged)
        self.assertEqual(self.prepared, [])
        self.assertEqual(self.consumed, [])

    def test_no_default_auth_or_ledger_callback(self):
        for name in ("verify_grant", "prepare", "consume_reservation"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.adapter(**{name: None})

    def test_public_tenant_metadata_rejected_but_state_data_allowed(self):
        for key in ("tenant_id", "metadata", "key_id", "reservation_id"):
            raw = json.dumps(dict(self.request, **{key: "spoof"})).encode()
            with self.subTest(key=key), self.assertRaises(AdmissionRejected):
                self.adapter().admit(raw, b"platform-authenticated-envelope")
        self.assertEqual(self.verified, [])
        self.admit()

    def test_exact_bytes_binding_including_whitespace(self):
        with self.assertRaisesRegex(AdmissionRejected, "payload binding"):
            self.adapter().admit(self.raw + b" ", b"platform-authenticated-envelope")
        self.assertEqual(self.prepared, [])

    def test_wrong_runtime_revision_policy_and_model_fail_closed(self):
        for key, value in (("runtime_generation", "other"), ("checkpoint_revision", "b" * 40),
                           ("policy_revision", "old"), ("model", "other")):
            saved = self.claims[key]
            self.claims[key] = value
            with self.subTest(key=key), self.assertRaises(AdmissionRejected):
                self.admit()
            self.claims[key] = saved
        self.assertEqual(self.prepared, [])

    def test_deadline_and_future_issue_time(self):
        for key, value in (("deadline_unix_ms", 10000), ("issued_at_unix_ms", 10001)):
            saved = self.claims[key]
            self.claims[key] = value
            with self.subTest(key=key), self.assertRaises(AdmissionRejected):
                self.admit()
            self.claims[key] = saved
        self.assertEqual(self.prepared, [])

    def test_malformed_claims_rejected_after_verifier(self):
        del self.claims["reservation_id"]
        with self.assertRaisesRegex(AdmissionRejected, "verification"):
            self.admit()
        self.assertEqual(len(self.verified), 1)
        self.assertEqual(self.prepared, [])

    def test_duplicate_nonfinite_utf8_size_and_depth_rejected(self):
        bad = [b'{"model":"a","model":"b"}', b'{"state":NaN}',
               b'{"state":1e999}', b'\xff', b' ' * (2 * 1024 * 1024 + 1),
               b'{"state":' + b'[' * 33 + b'0' + b']' * 33 + b'}',
               b'{"state":{"x":1,"x":2}}']
        for raw in bad:
            with self.subTest(length=len(raw)), self.assertRaises(AdmissionRejected):
                self.adapter().admit(raw, b"platform-authenticated-envelope")
        self.assertEqual(self.verified, [])

    def test_braces_in_strings_do_not_count_as_depth(self):
        self.request["state"] = "[" * 100 + '\\"' + "]" * 100
        self.raw = json.dumps(self.request).encode()
        self.claims["request_sha256"] = hashlib.sha256(self.raw).hexdigest()
        self.admit()

    def test_prepared_limits_and_types_enforced_before_consume(self):
        for rows, tokens in ((5, 1), (1, 101), (0, 1), (1, 0), (True, 1), (1, 1.5)):
            with self.subTest(rows=rows, tokens=tokens), self.assertRaises(AdmissionRejected):
                self.admit(prepare=lambda request: PreparedInput(rows, tokens, b"payload"))
        with self.assertRaises(AdmissionRejected):
            self.admit(max_encoded_tokens=10)
        self.assertEqual(self.consumed, [])

    def test_preparation_expiry_does_not_consume(self):
        def slow_prepare(request):
            self.mono = 52.
            return PreparedInput(1, 1, b"payload")
        with self.assertRaisesRegex(AdmissionRejected, "expired"):
            self.admit(prepare=slow_prepare)
        self.assertEqual(self.consumed, [])

    def test_replay_is_rejected_by_required_platform_ledger(self):
        adapter = self.adapter()
        adapter.admit(self.raw, b"platform-authenticated-envelope")
        with self.assertRaisesRegex(AdmissionRejected, "already consumed"):
            adapter.admit(self.raw, b"platform-authenticated-envelope")
        self.assertEqual(len(self.consumed), 2)

    def test_ledger_failure_and_non_true_result_fail_closed(self):
        def unavailable(*args):
            raise RuntimeError("private database endpoint")
        for consume in (unavailable, lambda *args: None, lambda *args: 1):
            with self.subTest(callback=consume), self.assertRaises(AdmissionRejected):
                self.admit(consume_reservation=consume)


if __name__ == "__main__":
    unittest.main()
