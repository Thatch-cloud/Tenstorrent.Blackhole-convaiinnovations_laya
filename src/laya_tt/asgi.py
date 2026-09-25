"""Internal ASGI seam. Hosting and authenticated grant issuance stay platform-owned."""
from __future__ import annotations

import asyncio
import base64
import binascii
from concurrent.futures import CancelledError as FutureCancelled
import json
import threading

from .admission import AdmissionRejected, InvalidRequest, PreparationFailed, RequestTooLarge
from .ledger import JournalReplay
from .service import AdmissionCapacityExceeded, ServiceNotReady
from .worker import DeadlineExceeded, DuplicateRequest, QueueFull, WorkerClosed


GRANT_HEADER = b"x-thatch-admission-grant"
MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_GRANT_HEADER_BYTES = 16384


class _TransportError(Exception):
    def __init__(self, status, code):
        self.status, self.code = status, code


class DecisionASGI:
    """Inject a configured DecisionService; this class never creates/listens itself.

    The internal header carries canonical standard-base64 opaque grant bytes, not
    a trusted tenant field. The service verifier must authenticate those bytes.
    ``max_pending`` bounds body readers, submission threads and waiting responses.
    """
    def __init__(self, service, *, max_pending=16, body_timeout_seconds=30):
        if type(max_pending) is not int or max_pending <= 0:
            raise ValueError("max_pending must be a positive integer")
        if not isinstance(body_timeout_seconds, (int, float)) or isinstance(body_timeout_seconds, bool) or not 0 < body_timeout_seconds < float("inf"):
            raise ValueError("body timeout must be positive and finite")
        self._service = service
        self._slots = threading.BoundedSemaphore(max_pending)
        self._body_timeout = body_timeout_seconds

    @staticmethod
    async def _send(send, status, body, extra_headers=()):
        encoded = json.dumps(body, allow_nan=False, separators=(",", ":")).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(encoded)).encode()),
                                (b"cache-control", b"no-store"), *extra_headers]})
        await send({"type": "http.response.body", "body": encoded})

    @classmethod
    async def _error(cls, send, status, code, extra_headers=()):
        await cls._send(send, status, {"error": {"code": code}}, extra_headers)

    @staticmethod
    def _headers(scope):
        found = {}
        selected = {GRANT_HEADER, b"content-type", b"content-length"}
        for key, value in scope.get("headers", []):
            key = key.lower()
            if key in selected:
                if key in found:
                    raise _TransportError(400, "duplicate_header")
                found[key] = value
        media = found.get(b"content-type", b"").lower().replace(b" ", b"")
        if media not in (b"application/json", b"application/json;charset=utf-8"):
            raise _TransportError(415, "unsupported_media_type")
        value = found.get(GRANT_HEADER, b"")
        if not value or len(value) > MAX_GRANT_HEADER_BYTES:
            raise _TransportError(400, "invalid_admission_header")
        try:
            envelope = base64.b64decode(value, validate=True)
            if not envelope or base64.b64encode(envelope) != value:
                raise ValueError("noncanonical base64")
        except (ValueError, binascii.Error):
            raise _TransportError(400, "invalid_admission_header") from None
        length = found.get(b"content-length")
        if length is not None:
            if not length or len(length) > 10 or not length.isdigit():
                raise _TransportError(400, "invalid_content_length")
            length = int(length)
            if length > MAX_BODY_BYTES:
                raise _TransportError(413, "request_too_large")
        return envelope, length

    async def _body(self, receive, expected_length):
        body = bytearray()
        async with asyncio.timeout(self._body_timeout):
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return None
                if message["type"] != "http.request":
                    raise _TransportError(400, "invalid_request_event")
                chunk = message.get("body", b"")
                if not isinstance(chunk, bytes):
                    raise _TransportError(400, "invalid_body")
                if len(body) + len(chunk) > MAX_BODY_BYTES:
                    raise _TransportError(413, "request_too_large")
                body.extend(chunk)
                if not message.get("more_body", False):
                    break
        if expected_length is not None and len(body) != expected_length:
            raise _TransportError(400, "content_length_mismatch")
        return bytes(body)

    @staticmethod
    async def _disconnect(receive):
        while True:
            try:
                message = await receive()
            except Exception:
                return
            if message["type"] == "http.disconnect":
                return
            # The body has ended. Extra body events violate the ASGI request;
            # stop delivery rather than admitting another request on this scope.
            if message["type"] == "http.request":
                return

    @staticmethod
    def _failure(exc):
        if isinstance(exc, (AdmissionCapacityExceeded, QueueFull)):
            return 429, "capacity_exceeded"
        if isinstance(exc, RequestTooLarge):
            return 413, "request_too_large"
        if isinstance(exc, InvalidRequest):
            return 400, "invalid_request"
        if isinstance(exc, PreparationFailed):
            return 500, "preparation_failed"
        if isinstance(exc, AdmissionRejected):
            return 403, "invalid_admission"
        if isinstance(exc, DeadlineExceeded):
            return 504, "deadline_exceeded"
        if isinstance(exc, (DuplicateRequest, JournalReplay)):
            return 409, "duplicate_request"
        if isinstance(exc, (ServiceNotReady, WorkerClosed, FutureCancelled)):
            return 503, "service_unavailable"
        return 500, "execution_failed"

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            raise ValueError("DecisionASGI supports HTTP scopes only")
        path, method = scope.get("path"), scope.get("method")
        if path in ("/healthz", "/readyz"):
            if method != "GET":
                return await self._error(send, 405, "method_not_allowed", [(b"allow", b"GET")])
            readiness = self._service.readiness()
            status = 200 if path == "/healthz" or readiness["ready"] else 503
            return await self._send(send, status, readiness)
        if path != "/v1/decisions":
            return await self._error(send, 404, "not_found")
        if method != "POST":
            return await self._error(send, 405, "method_not_allowed", [(b"allow", b"POST")])
        if not self._slots.acquire(blocking=False):
            return await self._error(send, 429, "capacity_exceeded")
        admission_task = disconnected = wrapped = None
        future = None
        release_here = True
        sending = False
        try:
            envelope, expected_length = self._headers(scope)
            try:
                raw = await self._body(receive, expected_length)
            except TimeoutError:
                raise _TransportError(408, "request_body_timeout") from None
            if raw is None:
                return
            disconnected = asyncio.create_task(self._disconnect(receive))
            admission_task = asyncio.create_task(asyncio.to_thread(self._service.submit, raw, envelope))
            done, _ = await asyncio.wait((admission_task, disconnected), return_when=asyncio.FIRST_COMPLETED)
            if disconnected in done:
                # Python threads cannot be preempted. Retain this bounded slot
                # until preparation returns, then cancel delivery immediately.
                try:
                    future = await asyncio.shield(admission_task)
                    future.cancel()
                except Exception:
                    pass
                return
            future = admission_task.result()
            wrapped = asyncio.wrap_future(future)
            done, _ = await asyncio.wait((wrapped, disconnected), return_when=asyncio.FIRST_COMPLETED)
            if disconnected in done:
                future.cancel()
                if wrapped.done() and not wrapped.cancelled():
                    wrapped.exception()  # Retrieve any simultaneously completed failure.
                else:
                    wrapped.cancel()
                return
            if wrapped.cancelled():
                raise FutureCancelled()
            response = wrapped.result()
            sending = True
            return await self._send(send, 200, response)
        except asyncio.CancelledError:
            if future is not None:
                future.cancel()
            elif admission_task is not None:
                # Server-side task cancellation does not stop the submission
                # thread. Own its eventual Future and slot until it completes.
                release_here = False
                def submitted(task):
                    try:
                        task.result().cancel()
                    except BaseException:
                        pass
                    finally:
                        self._slots.release()
                admission_task.add_done_callback(submitted)
            raise
        except _TransportError as exc:
            return await self._error(send, exc.status, exc.code)
        except Exception as exc:
            if sending:
                return  # Transport failed during delivery; never start a second response.
            if disconnected is not None and disconnected.done():
                return
            status, code = self._failure(exc)
            return await self._error(send, status, code)
        finally:
            if disconnected is not None:
                disconnected.cancel()
            if wrapped is not None and not wrapped.done():
                wrapped.cancel()
            if release_here:
                self._slots.release()
