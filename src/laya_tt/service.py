"""Internal decision service: verified admission, serialized execution, response.

No transport/authentication implementation or durable usage ledger lives here.
Bootstrap supplies the real backend, verified runtime identity and startup test.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import threading
import time
from typing import Callable

from jsonschema import Draft202012Validator

from .admission import AdmissionAdapter, AdmittedRequest
from .contracts import load_schema
from .reference import validate_answers
from .worker import SerializedWorker


class ServiceNotReady(RuntimeError):
    """The service is starting, draining, stopped, or failed."""


class AdmissionCapacityExceeded(RuntimeError):
    """The bounded preparation/admission pool is full; no grant was consumed."""


class InvalidBackendResult(RuntimeError):
    """Backend output failed the decision response contract."""


@dataclass(frozen=True)
class RuntimeIdentity:
    checkpoint_revision: str
    runtime_revision: str
    runtime_generation: str
    backend: str
    model: str = "laya-english"

    def __post_init__(self):
        if (not re.fullmatch(r"[0-9a-f]{40}", self.checkpoint_revision)
                or not 1 <= len(self.runtime_revision) <= 128
                or not 1 <= len(self.runtime_generation) <= 128
                or self.backend not in ("cpu-reference", "tt-blackhole")
                or self.model != "laya-english"):
            raise ValueError("invalid observed runtime identity")


@dataclass(frozen=True)
class _QueuedDecision:
    admitted: AdmittedRequest
    queued_at: float


class DecisionService:
    """Compose existing platform callbacks without supplying permissive defaults.

    RuntimeIdentity must describe the backend actually loaded by trusted startup;
    setting its label cannot create a TT backend or prove hardware acceptance.
    The backend is never replaced/retried on another device after errors.
    """
    def __init__(self, *, admission: AdmissionAdapter, backend: Callable,
                 runtime: RuntimeIdentity, schema_dir: Path | None = None, worker_limits=None,
                 max_admitting: int = 8):
        if type(max_admitting) is not int or max_admitting <= 0:
            raise ValueError("max_admitting must be a positive integer")
        self._admission, self._backend, self.runtime = admission, backend, runtime
        schema = (load_schema("decision-response") if schema_dir is None else
                  json.loads((Path(schema_dir) / "decision-response.schema.json").read_text(encoding="utf-8")))
        Draft202012Validator.check_schema(schema)
        self._validator = Draft202012Validator(schema)
        self._cv = threading.Condition()
        self._state = "starting"
        self._booting = False
        self._runtime_failed = False
        self._admitting = 0
        self._max_admitting = max_admitting
        self._worker = SerializedWorker(self._execute, **(worker_limits or {}))

    def readiness(self) -> dict:
        """Process state only; ready requires successful explicit startup testing."""
        with self._cv:
            return {"state": self._state, "ready": self._state == "ready",
                    "runtime_generation": self.runtime.runtime_generation,
                    "backend": self.runtime.backend}

    def start(self, self_test: Callable[[], bool]) -> None:
        """Test the loaded model before admission; a TCP probe is insufficient.

        Callback runs once, must return exactly True, and must exercise the actual
        model/checkpoint. No default callback is provided. A failed start requires
        rebuilding this service generation rather than silently retrying devices.
        """
        if not callable(self_test):
            raise ValueError("a model self-test callback is required")
        with self._cv:
            if self._state != "starting" or self._booting:
                raise ServiceNotReady("startup is not available")
            self._booting = True
        try:
            if self_test() is not True:
                raise ServiceNotReady("model self-test failed")
            with self._cv:
                if self._state != "starting":
                    raise ServiceNotReady("service was drained during startup")
                self._state = "ready"
        except BaseException:
            with self._cv:
                if self._state == "starting":
                    self._state = "failed"
            raise
        finally:
            with self._cv:
                self._booting = False
                self._cv.notify_all()

    def submit(self, raw_request: bytes, authenticated_envelope: bytes):
        with self._cv:
            if self._state != "ready":
                raise ServiceNotReady("service is not ready for admission")
            if self._admitting >= self._max_admitting:
                raise AdmissionCapacityExceeded("request preparation capacity exhausted")
            self._admitting += 1
        try:
            admitted = self._admission.admit(raw_request, authenticated_envelope)
            context = admitted.context
            if any(context[key] != getattr(self.runtime, key) for key in
                   ("model", "checkpoint_revision", "runtime_generation")):
                raise ServiceNotReady("admission targets a different loaded runtime")
            with self._cv:
                if self._state != "ready":
                    raise ServiceNotReady("service drained during admission")
            # Do not nest the service and worker locks: Future callbacks may
            # inspect readiness while the worker's deadline monitor holds its
            # lock. Drain closes worker admission independently; an admission
            # already in progress either queues before that close or is refused.
            return self._worker.submit(
                context["tenant_id"], context["execution_attempt_id"],
                _QueuedDecision(admitted, time.monotonic()),
                cost=admitted.prepared.encoded_tokens, deadline=admitted.deadline)
        finally:
            with self._cv:
                self._admitting -= 1
                self._cv.notify_all()

    def _execute(self, work: _QueuedDecision) -> dict:
        with self._cv:
            if self._runtime_failed:
                raise ServiceNotReady("runtime failed; a new service generation is required")
        started = time.monotonic()
        admitted = work.admitted
        try:
            result = self._backend(admitted.prepared.payload)
        except BaseException:
            self._mark_failed()
            raise
        finished = time.monotonic()
        try:
            if type(result["encoded_tokens"]) is not int or result["encoded_tokens"] != admitted.prepared.encoded_tokens:
                raise ValueError("token accounting differs")
            # Serialize to own the response and reject NaN/Infinity before schema
            # comparison. No backend-owned mutable answer object escapes.
            answers = json.loads(json.dumps(result["answers"], allow_nan=False))
            if list(answers) != list(admitted.request["questions"]):
                raise ValueError("question identity or order differs")
            validate_answers({"answers": answers}, admitted.request["questions"])
            response = {
                "schema_version": "1", "request_id": admitted.context["request_id"],
                "model": self.runtime.model,
                "checkpoint_revision": self.runtime.checkpoint_revision,
                "runtime_revision": self.runtime.runtime_revision,
                "runtime_generation": self.runtime.runtime_generation,
                "backend": self.runtime.backend, "answers": answers,
                "usage": {"requests": 1, "question_rows": admitted.prepared.question_rows,
                          "encoded_tokens": admitted.prepared.encoded_tokens, "output_tokens": 0,
                          "queue_ms": max(0, int((started - work.queued_at) * 1000)),
                          "execution_ms": max(0, int((finished - started) * 1000))},
            }
            self._validator.validate(response)
            return response
        except Exception:
            self._mark_failed()
            raise InvalidBackendResult("backend result violates decision contract") from None

    def _mark_failed(self):
        with self._cv:
            self._runtime_failed = True
            if self._state == "ready":
                self._state = "failed"
            self._cv.notify_all()

    def drain(self, *, wait: bool = True, cancel_queued: bool = False,
              timeout: float | None = None) -> bool:
        """Close admission and wait for preparation, self-test and model ownership.

        False never authorizes backend destruction/card reassignment. Outstanding
        reservation reconciliation and durable usage remain platform-owned.
        """
        end = None if timeout is None else time.monotonic() + max(0, timeout)
        with self._cv:
            if self._state == "stopped":
                return True
            self._state = "draining"
        stopped = self._worker.shutdown(wait=wait, cancel_queued=cancel_queued, timeout=timeout)
        with self._cv:
            while wait and (self._admitting or self._booting):
                remaining = None if end is None else end - time.monotonic()
                if remaining is not None and remaining <= 0:
                    break
                self._cv.wait(remaining)
            if stopped and not self._admitting and not self._booting:
                self._state = "stopped"
                return True
            return False
