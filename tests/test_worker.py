import threading
import time
import unittest
from concurrent.futures import CancelledError

from laya_tt.worker import (
    DeadlineExceeded, DuplicateRequest, QueueFull, SerializedWorker, WorkerClosed,
)


class WorkerTests(unittest.TestCase):
    def blocking_worker(self, **limits):
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def backend(payload):
            calls.append(payload)
            if payload == "hold":
                entered.set()
                if not release.wait(3):
                    raise RuntimeError("test backend was not released")
            return payload

        worker = SerializedWorker(backend, **limits)
        self.addCleanup(lambda: worker.shutdown(timeout=3))
        self.addCleanup(release.set)
        initial = worker.submit("blocker", "hold", "hold")
        self.assertTrue(entered.wait(1))
        return worker, release, calls, initial

    def test_tenant_fairness_and_same_request_ids(self):
        worker, release, calls, _ = self.blocking_worker()
        futures = [worker.submit("a", str(i), f"a{i}") for i in range(3)]
        futures += [worker.submit("b", str(i), f"b{i}") for i in range(2)]
        release.set()
        self.assertEqual([f.result(2) for f in futures], ["a0", "a1", "a2", "b0", "b1"])
        self.assertTrue(worker.shutdown(timeout=2))
        self.assertEqual(calls, ["hold", "a0", "b0", "a1", "b1", "a2"])

    def test_queued_cost_is_sum_and_cancel_reclaims_budget(self):
        worker, _, _, _ = self.blocking_worker(max_queued_cost=10, max_queued_cost_per_tenant=7)
        first = worker.submit("a", "1", "one", cost=4)
        worker.submit("a", "2", "two", cost=3)
        with self.assertRaises(QueueFull):
            worker.submit("a", "3", "three", cost=1)
        worker.submit("b", "1", "four", cost=3)
        with self.assertRaises(QueueFull):
            worker.submit("c", "1", "five", cost=1)
        self.assertTrue(first.cancel())
        worker.submit("c", "1", "five", cost=4)

    def test_count_limits(self):
        worker, _, _, _ = self.blocking_worker(max_queued=2, max_queued_per_tenant=1)
        worker.submit("a", "1", "one")
        with self.assertRaises(QueueFull):
            worker.submit("a", "2", "two")
        worker.submit("b", "1", "three")
        with self.assertRaises(QueueFull):
            worker.submit("c", "1", "four")

    def test_expired_request_never_enters_backend(self):
        worker, release, calls, _ = self.blocking_worker()
        expired = worker.submit("a", "1", "expired", deadline=time.monotonic() - 1)
        queued = worker.submit("a", "2", "expires-queued", deadline=time.monotonic() + .03)
        for future in (expired, queued):
            with self.assertRaises(DeadlineExceeded):
                future.result(1)
        release.set()
        self.assertTrue(worker.shutdown(timeout=2))
        self.assertEqual(calls, ["hold"])

    def test_cancel_is_scoped_to_tenant(self):
        worker, release, calls, _ = self.blocking_worker()
        a = worker.submit("a", "same", "a")
        b = worker.submit("b", "same", "b")
        self.assertTrue(worker.cancel("a", "same"))
        self.assertFalse(worker.cancel("other", "same"))
        release.set()
        with self.assertRaises(CancelledError):
            a.result(1)
        self.assertEqual(b.result(1), "b")
        self.assertEqual(calls, ["hold", "b"])

    def test_inflight_cancel_retains_backend_and_id_until_release(self):
        worker, release, calls, active = self.blocking_worker()
        next_result = worker.submit("b", "next", "next")
        self.assertTrue(active.cancel())
        with self.assertRaises(DuplicateRequest):
            worker.submit("blocker", "hold", "replacement")
        self.assertFalse(worker.shutdown(timeout=.02))
        self.assertFalse(next_result.done())
        self.assertEqual(calls, ["hold"])
        release.set()
        self.assertTrue(worker.shutdown(timeout=2))
        self.assertEqual(next_result.result(), "next")

    def test_inflight_deadline_delivers_before_backend_returns(self):
        entered, release = threading.Event(), threading.Event()
        def backend(payload):
            entered.set()
            release.wait(2)
            return payload
        worker = SerializedWorker(backend)
        self.addCleanup(lambda: worker.shutdown(timeout=3))
        self.addCleanup(release.set)
        future = worker.submit("a", "1", object(), deadline=time.monotonic() + .05)
        self.assertTrue(entered.wait(1))
        with self.assertRaises(DeadlineExceeded):
            future.result(1)
        self.assertFalse(worker.shutdown(timeout=.01))
        release.set()
        self.assertTrue(worker.shutdown(timeout=2))

    def test_shutdown_cancels_queue_and_rejects_new_work(self):
        worker, release, calls, active = self.blocking_worker()
        queued = worker.submit("a", "1", "queued")
        self.assertFalse(worker.shutdown(wait=False, cancel_queued=True))
        self.assertTrue(queued.cancelled())
        self.assertFalse(active.done())
        with self.assertRaises(WorkerClosed):
            worker.submit("b", "1", "rejected")
        release.set()
        self.assertTrue(worker.shutdown(timeout=2))
        self.assertEqual(calls, ["hold"])

    def test_errors_do_not_kill_worker(self):
        def backend(payload):
            if payload == "bad":
                raise ValueError("backend failure")
            return payload
        worker = SerializedWorker(backend)
        self.addCleanup(worker.shutdown)
        with self.assertRaises(ValueError):
            worker.submit("a", "1", "bad").result(1)
        self.assertEqual(worker.submit("b", "1", "good").result(1), "good")

    def test_backend_never_concurrent_under_parallel_submission(self):
        lock = threading.Lock()
        active = 0
        peak = 0
        def backend(payload):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(.002)
            with lock:
                active -= 1
            return payload
        worker = SerializedWorker(backend)
        self.addCleanup(worker.shutdown)
        futures = []
        def submit(tenant):
            for i in range(10):
                futures.append(worker.submit(tenant, str(i), (tenant, i)))
        threads = [threading.Thread(target=submit, args=(str(i),)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(len({f.result(3) for f in futures}), 40)
        self.assertEqual(peak, 1)


    def test_success_callback_shutdown_does_not_join_its_peer_thread(self):
        worker, release, _, future = self.blocking_worker()
        callback_done = threading.Event()
        outcomes = []
        def completed(_future):
            outcomes.append(worker.shutdown(wait=True))
            callback_done.set()
        future.add_done_callback(completed)
        release.set()
        self.assertTrue(callback_done.wait(1), "shutdown deadlocked in completion callback")
        self.assertEqual(outcomes, [False])
        self.assertEqual(future.result(), "hold")
        self.assertTrue(worker.shutdown(timeout=2))

    def test_expiry_callback_shutdown_retains_backend_without_joining_worker(self):
        entered, release = threading.Event(), threading.Event()
        def backend(payload):
            entered.set()
            release.wait(3)
            return payload
        worker = SerializedWorker(backend)
        self.addCleanup(lambda: worker.shutdown(timeout=3))
        self.addCleanup(release.set)
        future = worker.submit("tenant", "attempt", "payload", deadline=time.monotonic() + .15)
        self.assertTrue(entered.wait(1))
        callback_done = threading.Event()
        outcomes = []
        def expired(_future):
            outcomes.append(worker.shutdown(wait=True))
            callback_done.set()
        future.add_done_callback(expired)
        self.assertTrue(callback_done.wait(1), "shutdown deadlocked in deadline callback")
        self.assertEqual(outcomes, [False])
        with self.assertRaises(DeadlineExceeded):
            future.result()
        self.assertFalse(worker.shutdown(timeout=.01), "backend ownership was released early")
        release.set()
        self.assertTrue(worker.shutdown(timeout=2))


    def test_cancel_callback_runs_without_worker_lock_and_nested_drain_returns_false(self):
        worker, release, calls, _ = self.blocking_worker()
        future = worker.submit("a", "cancel", "must-not-execute")
        observed = []
        def cancelled(_future):
            other_done = threading.Event()
            def inspect():
                worker.shutdown(wait=False)
                other_done.set()
            thread = threading.Thread(target=inspect, daemon=True)
            thread.start()
            observed.append(other_done.wait(.5))
            observed.append(worker.shutdown(wait=True))
        future.add_done_callback(cancelled)
        self.assertTrue(worker.cancel("a", "cancel"))
        self.assertEqual(observed, [True, False])
        release.set()
        self.assertTrue(worker.shutdown(timeout=2))
        self.assertEqual(calls, ["hold"])

    def test_shutdown_detaches_queued_work_before_reentrant_cancellation_callbacks(self):
        worker, release, calls, _ = self.blocking_worker()
        first = worker.submit("a", "1", "must-not-execute-1")
        second = worker.submit("b", "2", "must-not-execute-2")
        observed = []
        def cancelled(_future):
            release.set()
            observed.append(worker.shutdown(wait=True, cancel_queued=True))
        first.add_done_callback(cancelled)
        self.assertTrue(worker.shutdown(cancel_queued=True, timeout=2))
        self.assertEqual(observed, [False])
        self.assertTrue(first.cancelled())
        self.assertTrue(second.cancelled())
        self.assertEqual(calls, ["hold"])

    def test_expiry_callback_does_not_hold_worker_lock(self):
        worker, release, _, _ = self.blocking_worker()
        future = worker.submit("a", "expires", "unused", deadline=time.monotonic() + .15)
        callback_done = threading.Event()
        observed = []
        def expired(_future):
            other_done = threading.Event()
            def inspect():
                worker.shutdown(wait=False)
                other_done.set()
            thread = threading.Thread(target=inspect, daemon=True)
            thread.start()
            observed.append(other_done.wait(.5))
            callback_done.set()
        future.add_done_callback(expired)
        self.assertTrue(callback_done.wait(1))
        self.assertEqual(observed, [True])
        release.set()
        self.assertTrue(worker.shutdown(timeout=2))

    def test_external_drain_waits_for_cancellation_callback_to_finish(self):
        worker, release, _, _ = self.blocking_worker()
        future = worker.submit("a", "cancel", "unused")
        callback_entered, callback_release = threading.Event(), threading.Event()
        self.addCleanup(callback_release.set)
        def cancelled(_future):
            callback_entered.set()
            callback_release.wait(2)
        future.add_done_callback(cancelled)
        cancelling = threading.Thread(target=lambda: worker.cancel("a", "cancel"), daemon=True)
        cancelling.start()
        self.assertTrue(callback_entered.wait(1))
        release.set()
        self.assertFalse(worker.shutdown(timeout=.03))
        callback_release.set()
        cancelling.join(1)
        self.assertFalse(cancelling.is_alive())
        self.assertTrue(worker.shutdown(timeout=2))


if __name__ == "__main__":
    unittest.main()
