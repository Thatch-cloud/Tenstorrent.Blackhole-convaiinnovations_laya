import asyncio
import base64
import hashlib
from concurrent.futures import Future
import json
import threading
import time
import unittest

from laya_tt.admission import AdmissionAdapter, AdmissionRejected, PreparedInput
from laya_tt.asgi import DecisionASGI, GRANT_HEADER, MAX_BODY_BYTES
from laya_tt.worker import SerializedWorker


class FakeService:
    def __init__(self):
        self.ready = True
        self.calls = []
        self.future = Future()
        self.future.set_result({"answers": {"q": "opaque transport test result"}})

    def readiness(self):
        return {"ready": self.ready, "state": "ready" if self.ready else "draining"}

    def submit(self, raw, envelope):
        self.calls.append((raw, envelope))
        return self.future


class ASGITests(unittest.IsolatedAsyncioTestCase):
    def scope(self, *, path="/v1/decisions", method="POST", headers=None):
        if headers is None:
            headers = [(b"content-type", b"application/json"),
                       (GRANT_HEADER, base64.b64encode(b"opaque-platform-grant"))]
        return {"type": "http", "path": path, "method": method, "headers": headers}

    async def invoke(self, app, scope=None, events=None):
        incoming = asyncio.Queue()
        for event in events or [{"type": "http.request", "body": b"{}"}]:
            await incoming.put(event)
        outgoing = []
        async def send(message):
            outgoing.append(message)
        await app(scope or self.scope(), incoming.get, send)
        return outgoing

    def error(self, messages):
        return messages[0]["status"], json.loads(messages[1]["body"])["error"]["code"]

    async def test_fragmented_body_passes_exact_bytes_and_opaque_envelope(self):
        service = FakeService()
        response = await self.invoke(DecisionASGI(service), events=[
            {"type": "http.request", "body": b'{"state":', "more_body": True},
            {"type": "http.request", "body": b'"value"}'}])
        self.assertEqual(response[0]["status"], 200)
        self.assertEqual(service.calls, [(b'{"state":"value"}', b"opaque-platform-grant")])
        self.assertIn((b"cache-control", b"no-store"), response[0]["headers"])

    async def test_chunked_oversize_rejected_without_service(self):
        service = FakeService()
        response = await self.invoke(DecisionASGI(service), events=[
            {"type": "http.request", "body": b"x" * MAX_BODY_BYTES, "more_body": True},
            {"type": "http.request", "body": b"x"}])
        self.assertEqual(self.error(response), (413, "request_too_large"))
        self.assertEqual(service.calls, [])

    async def test_bad_duplicate_or_large_envelope_rejected(self):
        for grant in (b"!!!", b"", b"eA", b"a" * 16385):
            service = FakeService()
            scope = self.scope(headers=[(b"content-type", b"application/json"), (GRANT_HEADER, grant)])
            response = await self.invoke(DecisionASGI(service), scope)
            self.assertEqual(self.error(response), (400, "invalid_admission_header"))
            self.assertEqual(service.calls, [])
        headers = self.scope()["headers"]
        response = await self.invoke(DecisionASGI(FakeService()), self.scope(headers=headers + [headers[-1]]))
        self.assertEqual(self.error(response), (400, "duplicate_header"))

    async def test_length_media_method_and_unknown_path(self):
        cases = [(self.scope(method="GET"), (405, "method_not_allowed")),
                 (self.scope(path="/other"), (404, "not_found")),
                 (self.scope(headers=[(b"content-type", b"text/plain")]), (415, "unsupported_media_type")),
                 (self.scope(headers=self.scope()["headers"]+[(b"content-length", b"3")]), (400, "content_length_mismatch")),
                 (self.scope(headers=self.scope()["headers"]+[(b"content-length", b"-1")]), (400, "invalid_content_length"))]
        for scope, expected in cases:
            self.assertEqual(self.error(await self.invoke(DecisionASGI(FakeService()), scope)), expected)

    async def test_readiness_withdrawn_while_health_remains_live(self):
        service = FakeService()
        app = DecisionASGI(service)
        ready = await self.invoke(app, self.scope(path="/readyz", method="GET"))
        self.assertEqual(ready[0]["status"], 200)
        service.ready = False
        ready = await self.invoke(app, self.scope(path="/readyz", method="GET"))
        live = await self.invoke(app, self.scope(path="/healthz", method="GET"))
        self.assertEqual(ready[0]["status"], 503)
        self.assertEqual(live[0]["status"], 200)
        self.assertFalse(json.loads(ready[1]["body"])["ready"])

    async def test_errors_are_machine_readable_and_sanitized(self):
        for error, expected in ((AdmissionRejected("SECRET CLAIM"), (403, "invalid_admission")),
                                (RuntimeError("SECRET PROMPT"), (500, "execution_failed"))):
            service = FakeService()
            def reject(*args, error=error):
                raise error
            service.submit = reject
            response = await self.invoke(DecisionASGI(service))
            self.assertEqual(self.error(response), expected)
            self.assertNotIn(b"SECRET", response[1]["body"])

    async def test_disconnect_cancels_delivery_but_retains_worker_ownership(self):
        started, release = threading.Event(), threading.Event()
        def backend(payload):
            started.set()
            release.wait(3)
            return {"answers": {}}
        worker = SerializedWorker(backend)
        service = FakeService()
        futures = []
        def submit(*args):
            future = worker.submit("verified-tenant", "attempt", b"owned")
            futures.append(future)
            return future
        service.submit = submit
        incoming, outgoing = asyncio.Queue(), []
        await incoming.put({"type": "http.request", "body": b"{}"})
        async def send(message):
            outgoing.append(message)
        task = asyncio.create_task(DecisionASGI(service)(self.scope(), incoming.get, send))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            await incoming.put({"type": "http.disconnect"})
            await asyncio.wait_for(task, 1)
            self.assertTrue(futures[0].cancelled())
            self.assertFalse(worker.shutdown(wait=False))
            self.assertEqual(outgoing, [])
        finally:
            release.set()
            await asyncio.to_thread(worker.shutdown, timeout=2)

    async def test_submission_does_not_block_event_loop_and_transport_capacity_is_bounded(self):
        started, release = threading.Event(), threading.Event()
        service = FakeService()
        def submit(*args):
            started.set()
            release.wait(3)
            return service.future
        service.submit = submit
        app = DecisionASGI(service, max_pending=1)
        task = asyncio.create_task(self.invoke(app))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            # This response requires a functioning event loop while synchronous
            # service preparation is blocked in a separate thread.
            response = await asyncio.wait_for(self.invoke(app), 1)
            self.assertEqual(self.error(response), (429, "capacity_exceeded"))
        finally:
            release.set()
            await asyncio.wait_for(task, 2)

    async def test_server_task_cancellation_owns_eventual_submission_future(self):
        started, release = threading.Event(), threading.Event()
        service = FakeService()
        service.future = Future()
        def submit(*args):
            started.set()
            release.wait(3)
            return service.future
        service.submit = submit
        app = DecisionASGI(service, max_pending=1)
        task = asyncio.create_task(self.invoke(app))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 1))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            response = await self.invoke(app)
            self.assertEqual(self.error(response), (429, "capacity_exceeded"))
        finally:
            release.set()
        async with asyncio.timeout(2):
            while not service.future.cancelled():
                await asyncio.sleep(.005)

    async def test_body_timeout_without_submitting(self):
        service = FakeService()
        incoming = asyncio.Queue()
        outgoing = []
        async def send(message):
            outgoing.append(message)
        await DecisionASGI(service, body_timeout_seconds=.01)(self.scope(), incoming.get, send)
        self.assertEqual(self.error(outgoing), (408, "request_body_timeout"))
        self.assertEqual(service.calls, [])

    def admission_service(self, raw, *, prepare=None, reject_grant=False):
        service = FakeService()
        now = int(time.time() * 1000)
        claims = dict(schema_version="1", request_id="request", execution_attempt_id="attempt",
            usage_event_id="usage", tenant_id="verified-tenant", key_id="key",
            runtime_generation="generation", policy_revision="policy", reservation_id="reservation",
            model="laya-english", checkpoint_revision="a" * 40,
            issued_at_unix_ms=now-100, deadline_unix_ms=now+5000,
            max_question_rows=1, max_encoded_tokens=10,
            request_sha256=hashlib.sha256(raw).hexdigest())
        service.verifier_calls = 0
        service.reservation_calls = 0
        def verify(envelope):
            service.verifier_calls += 1
            if reject_grant or envelope != b"opaque-platform-grant":
                raise PermissionError("secret credential detail")
            return claims
        def consume(*args):
            service.reservation_calls += 1
            return True
        adapter = AdmissionAdapter(verify_grant=verify,
            prepare=prepare or (lambda request: PreparedInput(1, 5, request)),
            consume_reservation=consume, checkpoint_revision="a" * 40,
            runtime_generation="generation", policy_revision="policy")
        def submit(request, envelope):
            adapter.admit(request, envelope)
            return service.future
        service.submit = submit
        return service

    def native_request(self):
        return {"model": "laya-english", "state": "state", "questions": {
            "q": {"type": "noul", "instructions": "Evaluate"}}}

    async def test_actual_admission_malformed_json_and_schema_are_400(self):
        for raw in (b'{broken', b'{"model":"laya-english"}',
                    json.dumps(dict(self.native_request(), tenant_id="spoof")).encode()):
            service = self.admission_service(raw)
            response = await self.invoke(DecisionASGI(service), events=[{"type": "http.request", "body": raw}])
            self.assertEqual(self.error(response), (400, "invalid_request"))
            self.assertEqual(service.verifier_calls, 0)
            self.assertEqual(service.reservation_calls, 0)

    async def test_actual_admission_schema_and_prepared_overflow_are_413(self):
        request = self.native_request()
        request["questions"]["q"]["instructions"] = "x" * 4097
        raw = json.dumps(request).encode()
        service = self.admission_service(raw)
        response = await self.invoke(DecisionASGI(service), events=[{"type": "http.request", "body": raw}])
        self.assertEqual(self.error(response), (413, "request_too_large"))
        self.assertEqual(service.verifier_calls, 0)
        raw = json.dumps(self.native_request()).encode()
        service = self.admission_service(raw, prepare=lambda request: PreparedInput(1, 11, request))
        response = await self.invoke(DecisionASGI(service), events=[{"type": "http.request", "body": raw}])
        self.assertEqual(self.error(response), (413, "request_too_large"))
        self.assertEqual(service.verifier_calls, 1)
        self.assertEqual(service.reservation_calls, 0)

    async def test_actual_admission_grant_rejection_remains_403(self):
        raw = json.dumps(self.native_request()).encode()
        service = self.admission_service(raw, reject_grant=True)
        response = await self.invoke(DecisionASGI(service), events=[{"type": "http.request", "body": raw}])
        self.assertEqual(self.error(response), (403, "invalid_admission"))
        self.assertNotIn(b"secret", response[1]["body"])
        self.assertEqual(service.verifier_calls, 1)
        self.assertEqual(service.reservation_calls, 0)

    async def test_native_truncation_413_and_internal_preparation_failure_500(self):
        from laya_tt.cpu_backend import InputWouldTruncate
        raw = json.dumps(self.native_request()).encode()
        for failure, expected in ((InputWouldTruncate("sensitive option"), (413, "request_too_large")),
                                  (RuntimeError("sensitive implementation"), (500, "preparation_failed"))):
            def prepare(request, failure=failure):
                raise failure
            service = self.admission_service(raw, prepare=prepare)
            response = await self.invoke(DecisionASGI(service), events=[{"type": "http.request", "body": raw}])
            self.assertEqual(self.error(response), expected)
            self.assertNotIn(b"sensitive", response[1]["body"])
            self.assertEqual(service.reservation_calls, 0)


if __name__ == "__main__":
    unittest.main()
