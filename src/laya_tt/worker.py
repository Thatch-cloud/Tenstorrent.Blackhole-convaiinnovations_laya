"""Bounded, tenant-fair access to one non-reentrant inference backend."""

from collections import deque
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass
import threading
import time
from typing import Any, Callable


class QueueFull(RuntimeError):
    """An admission budget was exhausted."""


class DeadlineExceeded(TimeoutError):
    """The request's monotonic deadline elapsed."""


class WorkerClosed(RuntimeError):
    """The worker no longer admits requests."""


class DuplicateRequest(ValueError):
    """A tenant already has this request ID queued or executing."""


@dataclass
class _Work:
    tenant: str
    request_id: str
    payload: Any
    cost: int
    deadline: float | None
    future: Future


class SerializedWorker:
    """Round-robin tenant queues with bounded queued request and cost budgets.

    ``tenant_id`` is trusted internal context, never a public request field.
    The backend owns its input until it returns, even after cancellation.
    """

    def __init__(
        self,
        backend: Callable[[Any], Any],
        *,
        max_queued: int = 128,
        max_queued_per_tenant: int = 16,
        max_queued_cost: int = 65536,
        max_queued_cost_per_tenant: int = 8192,
    ) -> None:
        limits = (max_queued, max_queued_per_tenant,
                  max_queued_cost, max_queued_cost_per_tenant)
        if any(type(v) is not int or v <= 0 for v in limits):
            raise ValueError("queue limits must be positive integers")
        self._backend = backend
        self._limits = limits
        self._cv = threading.Condition(threading.RLock())
        self._queues: dict[str, deque[_Work]] = {}
        self._rotation: deque[str] = deque()
        self._pending: dict[tuple[str, str], _Work] = {}
        self._active: _Work | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="laya-worker", daemon=True)
        self._expiry = threading.Thread(target=self._expire, name="laya-deadlines", daemon=True)
        self._thread.start()
        self._expiry.start()

    def submit(self, tenant_id: str, request_id: str, payload: Any, *,
               cost: int = 1, deadline: float | None = None) -> Future:
        """Admit immutable/owned input; deadline is absolute ``time.monotonic``.

        Admission errors raise synchronously. Execution/expiry errors belong to
        the returned Future. IDs are unique within a tenant until backend release.
        """
        if not isinstance(tenant_id, str) or not tenant_id.strip():
            raise ValueError("verified tenant identity is required")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request identity is required")
        if type(cost) is not int or cost <= 0:
            raise ValueError("cost must be a positive integer")
        if deadline is not None:
            import math
            if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
                raise ValueError("deadline must be finite monotonic seconds")
        with self._cv:
            if self._closed:
                raise WorkerClosed("worker is draining")
            key = (tenant_id, request_id)
            if key in self._pending:
                raise DuplicateRequest("request identity is already active for this tenant")
            future: Future = Future()
            if deadline is not None and deadline <= time.monotonic():
                future.set_exception(DeadlineExceeded("request deadline elapsed"))
                return future
            queued = [item for queue in self._queues.values() for item in queue]
            own = self._queues.get(tenant_id, ())
            global_n, tenant_n, global_cost, tenant_cost = self._limits
            if (len(queued) >= global_n or len(own) >= tenant_n
                    or sum(item.cost for item in queued) + cost > global_cost
                    or sum(item.cost for item in own) + cost > tenant_cost):
                raise QueueFull("request queue budget exceeded")
            work = _Work(tenant_id, request_id, payload, cost, deadline, future)
            self._pending[key] = work
            if tenant_id not in self._queues:
                self._queues[tenant_id] = deque()
                self._rotation.append(tenant_id)
            self._queues[tenant_id].append(work)
            future.add_done_callback(lambda f: self._on_done(work))
            self._cv.notify_all()
            return future

    def _on_done(self, work: _Work) -> None:
        with self._cv:
            if work is not self._active:
                self._remove_queued(work)
            self._cv.notify_all()

    def _remove_queued(self, work: _Work) -> None:
        queue = self._queues.get(work.tenant)
        if queue is None:
            return
        try:
            queue.remove(work)
        except ValueError:
            return
        self._pending.pop((work.tenant, work.request_id), None)
        work.payload = None
        if not queue:
            del self._queues[work.tenant]
            self._rotation.remove(work.tenant)

    def cancel(self, tenant_id: str, request_id: str) -> bool:
        """Cancel delivery; an executing backend is not preempted or released."""
        with self._cv:
            work = self._pending.get((tenant_id, request_id))
            return work.future.cancel() if work is not None else False

    @staticmethod
    def _finish(future: Future, *, result: Any = None,
                error: BaseException | None = None) -> None:
        try:
            if error is None:
                future.set_result(result)
            else:
                future.set_exception(error)
        except InvalidStateError:
            pass  # Cancellation or the deadline monitor already won delivery.

    def _run(self) -> None:
        while True:
            with self._cv:
                self._cv.wait_for(lambda: bool(self._rotation) or self._closed)
                if not self._rotation:
                    return
                tenant = self._rotation.popleft()
                work = self._queues[tenant].popleft()
                if self._queues[tenant]:
                    self._rotation.append(tenant)
                else:
                    del self._queues[tenant]
                self._active = work
                self._cv.notify_all()
            try:
                if work.deadline is not None and work.deadline <= time.monotonic():
                    self._finish(work.future, error=DeadlineExceeded("request deadline elapsed"))
                elif not work.future.done():
                    result = self._backend(work.payload)
                    if work.deadline is not None and work.deadline <= time.monotonic():
                        self._finish(work.future, error=DeadlineExceeded("request deadline elapsed"))
                    else:
                        self._finish(work.future, result=result)
                    del result
            except BaseException as exc:
                self._finish(work.future, error=exc)
            finally:
                with self._cv:
                    self._pending.pop((work.tenant, work.request_id), None)
                    work.payload = None
                    self._active = None
                    self._cv.notify_all()

    def _expire(self) -> None:
        with self._cv:
            while True:
                if self._closed and not self._pending:
                    return
                now = time.monotonic()
                remaining = []
                for work in list(self._pending.values()):
                    if work.future.done() or work.deadline is None:
                        continue
                    if work.deadline <= now:
                        self._finish(work.future, error=DeadlineExceeded("request deadline elapsed"))
                    else:
                        remaining.append(work.deadline - now)
                self._cv.wait(timeout=min(remaining) if remaining else None)

    def shutdown(self, *, wait: bool = True, cancel_queued: bool = False,
                 timeout: float | None = None) -> bool:
        """Close admission and drain, or cancel queued work. Return true if stopped.

        Timeout never kills a backend or releases its device/buffers. The caller
        must retain the backend and allocation until a later successful drain.
        """
        with self._cv:
            self._closed = True
            if cancel_queued:
                for queue in list(self._queues.values()):
                    for work in list(queue):
                        work.future.cancel()
            self._cv.notify_all()
        if wait:
            end = None if timeout is None else time.monotonic() + max(0, timeout)
            for thread in (self._thread, self._expiry):
                if thread is threading.current_thread():
                    continue
                thread.join(None if end is None else max(0, end - time.monotonic()))
        return not self._thread.is_alive() and not self._expiry.is_alive()
