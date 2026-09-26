import asyncio
import threading

import pytest

from laya_tt.hosting import HostedRuntime
from laya_tt.ledger_client import LedgerUnavailable


class Application:
    def __init__(self, *, drain=True, delivery_error=None):
        self.events = []
        self.drain_result = drain
        self.delivery_error = delivery_error

    def start(self, callback):
        self.events.append("start")
        if callback() is not True:
            raise RuntimeError("sensitive checkpoint detail")

    def drain(self, **kwargs):
        assert kwargs["cancel_queued"] is True
        self.events.append("drain")
        return self.drain_result

    def deliver_pending(self, *, limit):
        assert limit == 8
        self.events.append("receipt")
        if self.delivery_error:
            raise self.delivery_error
        return 0

    async def asgi(self, scope, receive, send):
        await send({"type": "delegated", "path": scope["path"]})


async def lifecycle(host, *, after_start=None):
    incoming = asyncio.Queue()
    await incoming.put({"type": "lifespan.startup"})
    output = []

    async def send(message):
        output.append(message)
        if message["type"] == "lifespan.startup.complete":
            if after_start:
                await after_start(host)
            await incoming.put({"type": "lifespan.shutdown"})

    await host({"type": "lifespan"}, incoming.get, send)
    return output


def test_start_delegate_drain_and_final_receipt_batch():
    app = Application()
    host = HostedRuntime(app, self_test=lambda: True)
    async def request(host):
        sent = []
        async def send(message):
            sent.append(message)
        await host({"type": "http", "path": "/readyz"}, None, send)
        assert sent == [{"type": "delegated", "path": "/readyz"}]
    output = asyncio.run(lifecycle(host, after_start=request))
    assert [m["type"] for m in output] == ["lifespan.startup.complete", "lifespan.shutdown.complete"]
    assert app.events[0] == "start"
    assert app.events[-1] == "receipt"
    assert "drain" in app.events
    # A runtime generation is never implicitly restarted by another server loop.
    output = asyncio.run(lifecycle(host))
    assert output[0]["type"] == "lifespan.startup.failed"
    assert app.events.count("start") == 1


def test_failed_start_is_sanitized_and_drained():
    app = Application()
    host = HostedRuntime(app, self_test=lambda: False)
    output = asyncio.run(lifecycle(host))
    assert output == [{"type": "lifespan.startup.failed", "message": "runtime startup failed"}]
    assert app.events == ["start", "drain"]


def test_incomplete_drain_never_reports_complete():
    app = Application(drain=False)
    output = asyncio.run(lifecycle(HostedRuntime(app, self_test=lambda: True)))
    assert output[-1]["type"] == "lifespan.shutdown.failed"


def test_unavailable_delivery_retains_outbox_and_allows_clean_worker_shutdown():
    app = Application(delivery_error=LedgerUnavailable("private bridge detail"))
    output = asyncio.run(lifecycle(HostedRuntime(app, self_test=lambda: True)))
    assert output[-1]["type"] == "lifespan.shutdown.complete"


def test_unexpected_pump_failure_withdraws_http_and_reports_failed_shutdown():
    app = Application(delivery_error=RuntimeError("private journal detail"))
    async def after_start(host):
        for _ in range(1000):
            if host._delivery_failed:
                break
            await asyncio.sleep(.001)
        assert host._delivery_failed
        sent = []
        async def send(message):
            sent.append(message)
        await host({"type": "http", "path": "/readyz"}, None, send)
        assert sent[0]["status"] == 503
    output = asyncio.run(lifecycle(HostedRuntime(app, self_test=lambda: True), after_start=after_start))
    assert output[-1]["type"] == "lifespan.shutdown.failed"
    assert "private" not in str(output)


def test_cancellation_waits_for_blocking_operation_to_finish():
    async def run():
        host = HostedRuntime(Application(), self_test=lambda: True)
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        def operation():
            entered.set()
            assert release.wait(2)
            finished.set()
        task = asyncio.create_task(host._blocking(operation))
        while not entered.is_set():
            await asyncio.sleep(.001)
        task.cancel()
        await asyncio.sleep(.01)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
    asyncio.run(run())


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_lifecycle_timing_is_rejected(value):
    with pytest.raises(ValueError):
        HostedRuntime(Application(), self_test=lambda: True, receipt_interval=value)
