"""ASGI lifespan for one externally assigned, already loaded runtime generation."""
import asyncio
import math
import threading

from .ledger_client import LedgerUnavailable


class HostedRuntime:
    """Own startup, receipt scheduling and drain, never device allocation or release.

    Use one instance in one server process/event loop. The platform must establish
    ownership before constructing/loading RuntimeApplication. No restart, model
    fallback, credential loader or hardware teardown is supplied here.
    """

    def __init__(self, application, *, self_test, drain_timeout=30., receipt_interval=1.):
        if not callable(self_test):
            raise ValueError("loaded model self-test required")
        for value in (drain_timeout, receipt_interval):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("positive finite lifecycle timing required")
        self.application = application
        self.self_test = self_test
        self.drain_timeout = drain_timeout
        self.receipt_interval = receipt_interval
        self._claimed = threading.Lock()
        self._loop = None
        self._running = False
        self._stop = None
        self._delivery_failed = False

    async def _blocking(self, callback, **kwargs):
        # Cancelling an await must not abandon ownership of a blocking operation.
        task = asyncio.create_task(asyncio.to_thread(callback, **kwargs))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    async def _drain(self):
        try:
            return await self._blocking(self.application.drain,
                timeout=self.drain_timeout, cancel_queued=True) is True
        except Exception:
            return False

    async def _receipts(self):
        while not self._stop.is_set():
            try:
                await self._blocking(self.application.deliver_pending, limit=8)
            except LedgerUnavailable:
                # Original evidence remains in the journal. No receipt is relabelled
                # or acknowledged here; retry the next bounded batch after backoff.
                pass
            except Exception:
                self._delivery_failed = True
                self._running = False
                self._stop.set()
                await self._drain()
                return
            try:
                await asyncio.wait_for(self._stop.wait(), self.receipt_interval)
            except asyncio.TimeoutError:
                pass

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            if not self._running or asyncio.get_running_loop() is not self._loop:
                await send({"type": "http.response.start", "status": 503,
                            "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": b'{"error":"runtime_unavailable"}'})
                return
            return await self.application.asgi(scope, receive, send)
        if scope["type"] != "lifespan":
            raise ValueError("HTTP and lifespan scopes only")
        if not self._claimed.acquire(blocking=False):
            await send({"type": "lifespan.startup.failed", "message": "runtime lifecycle already claimed"})
            return
        # A generation cannot restart after stop/failure. Never release this claim.
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        pump = None
        drained = False
        try:
            message = await receive()
            if message["type"] != "lifespan.startup":
                drained = await self._drain()
                await send({"type": "lifespan.startup.failed", "message": "startup event required"})
                return
            try:
                await self._blocking(lambda: self.application.start(self.self_test))
            except Exception:
                drained = await self._drain()
                await send({"type": "lifespan.startup.failed", "message": "runtime startup failed"})
                return
            self._running = True
            pump = asyncio.create_task(self._receipts())
            await send({"type": "lifespan.startup.complete"})
            message = await receive()
            if message["type"] != "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.failed", "message": "shutdown event required"})
                return
            self._running = False
            drained = await self._drain()
            self._stop.set()
            await pump
            pump = None
            # Final bounded batch after the worker can no longer append receipts.
            if drained and not self._delivery_failed:
                try:
                    await self._blocking(self.application.deliver_pending, limit=8)
                except LedgerUnavailable:
                    pass  # Durable outbox retention is valid; acknowledgement is separate.
                except Exception:
                    self._delivery_failed = True
            complete = drained and not self._delivery_failed
            await send({"type": "lifespan.shutdown.complete" if complete else "lifespan.shutdown.failed",
                        **({} if complete else {"message": "runtime shutdown incomplete"})})
        finally:
            self._running = False
            self._stop.set()
            if not drained:
                await self._drain()
            if pump is not None:
                await pump
