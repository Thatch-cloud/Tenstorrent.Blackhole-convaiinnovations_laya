"""Fail-closed adapter boundary; platform authentication and ledger are injected."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


class AdmissionRejected(ValueError):
    """Safe rejection reason without request, credential, or verifier details."""


class InvalidRequest(AdmissionRejected):
    """Public request syntax or semantics are invalid (HTTP 400)."""


class RequestTooLarge(InvalidRequest):
    """Public or prepared request exceeds a size/token budget (HTTP 413)."""


class PreparationFailed(AdmissionRejected):
    """Trusted preparation failed internally; not a credential rejection."""


def _size_violation(error):
    return error.validator in {"maxLength", "maxItems", "maxProperties"} or any(
        _size_violation(child) for child in error.context)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidRequest("duplicate JSON key")
        result[key] = value
    return result


def _constant(_value):
    raise InvalidRequest("nonfinite JSON number")


def _parse_request(raw: bytes) -> dict:
    if type(raw) is not bytes:
        raise InvalidRequest("request must contain bytes")
    if len(raw) > 2 * 1024 * 1024:
        raise RequestTooLarge("request byte limit exceeded")
    try:
        text = raw.decode("utf-8", errors="strict")
        # Bound nesting before invoking the recursive JSON decoder. Braces
        # inside escaped strings are data, not structural nesting.
        depth = 0
        quoted = escaped = False
        for char in text:
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in "[{":
                depth += 1
                if depth > 32:
                    raise RequestTooLarge("request nesting limit exceeded")
            elif char in "]}":
                depth -= 1
        parsed = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
        stack = [parsed]
        while stack:
            item = stack.pop()
            if isinstance(item, float) and not math.isfinite(item):
                raise InvalidRequest("nonfinite JSON number")
            if isinstance(item, dict):
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
        if not isinstance(parsed, dict):
            raise InvalidRequest("request must be an object")
        return parsed
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        if isinstance(exc, AdmissionRejected):
            raise
        raise InvalidRequest("invalid request JSON") from None


@dataclass(frozen=True)
class PreparedInput:
    """Trusted tokenizer output; payload ownership transfers to the worker.

    JSON containers are copied and frozen on admission. Opaque tensor/buffer
    objects must be exclusively owned; the preparation callback must not retain
    mutable aliases. This wrapper is immutable, not a tensor write-protection API.
    """
    question_rows: int
    encoded_tokens: int
    payload: Any


@dataclass(frozen=True)
class AdmittedRequest:
    context: Mapping[str, Any]
    request: Mapping[str, Any]
    prepared: PreparedInput
    deadline: float


class AdmissionAdapter:
    """No default verifier, keys, bearer tokens, or in-process replay ledger.

    ``verify_grant`` authenticates opaque envelope bytes and returns verified
    claims. ``consume_reservation`` atomically checks/consumes the platform's
    durable reservation/replay record, returning exactly True on success.
    ``prepare`` must tokenize under CPU/resource bounds without executing a model.
    """

    def __init__(self, *, schema_dir: Path | None = None,
                 verify_grant: Callable[[bytes], Mapping[str, Any]],
                 prepare: Callable[[Mapping[str, Any]], PreparedInput],
                 consume_reservation: Callable[[Mapping[str, Any], int, int], bool],
                 checkpoint_revision: str, runtime_generation: str,
                 policy_revision: str, model: str = "laya-english",
                 max_question_rows: int = 64, max_encoded_tokens: int = 32768,
                 wall_time: Callable[[], float] = time.time,
                 monotonic: Callable[[], float] = time.monotonic):
        if not all(callable(callback) for callback in
                   (verify_grant, prepare, consume_reservation, wall_time, monotonic)):
            raise ValueError("platform verifier, preparation and ledger callbacks are required")
        if (type(max_question_rows) is not int or not 1 <= max_question_rows <= 64
                or type(max_encoded_tokens) is not int or not 1 <= max_encoded_tokens <= 32768):
            raise ValueError("invalid configured preparation limits")
        import re
        if (model != "laya-english" or not re.fullmatch(r"[0-9a-f]{40}", checkpoint_revision)
                or not runtime_generation or not policy_revision):
            raise ValueError("runtime identity must be explicitly configured")
        self._verify = verify_grant
        self._prepare = prepare
        self._consume = consume_reservation
        self._identity = dict(model=model, checkpoint_revision=checkpoint_revision,
                              runtime_generation=runtime_generation,
                              policy_revision=policy_revision)
        self._max_rows, self._max_tokens = max_question_rows, max_encoded_tokens
        self._wall, self._mono = wall_time, monotonic
        validators = []
        for name in ("decision-request", "admission-context"):
            from .contracts import load_schema
            schema = (load_schema(name) if schema_dir is None else
                      json.loads((Path(schema_dir) / f"{name}.schema.json").read_text(encoding="utf-8")))
            Draft202012Validator.check_schema(schema)
            validators.append(Draft202012Validator(schema))
        self._request_validator, self._context_validator = validators

    def _time_budget(self, context):
        now = self._wall() * 1000
        if not context["issued_at_unix_ms"] <= now < context["deadline_unix_ms"]:
            raise AdmissionRejected("grant not currently valid")
        return (context["deadline_unix_ms"] - now) / 1000

    def admit(self, raw_request: bytes, envelope: bytes) -> AdmittedRequest:
        request = _parse_request(raw_request)
        try:
            self._request_validator.validate(request)
        except ValidationError as exc:
            if _size_violation(exc):
                raise RequestTooLarge("request exceeds schema size limits") from None
            raise InvalidRequest("request does not match decision schema") from None
        if type(envelope) is not bytes:
            raise AdmissionRejected("opaque authenticated grant bytes required")
        try:
            claims = self._verify(envelope)
            if not isinstance(claims, Mapping):
                raise ValueError("claims must be a mapping")
            # Own a JSON snapshot, preventing verifier-side mutation after checks.
            context = json.loads(json.dumps(dict(claims), allow_nan=False))
            self._context_validator.validate(context)
        except Exception:
            raise AdmissionRejected("grant verification failed") from None
        if any(context[key] != value for key, value in self._identity.items()):
            raise AdmissionRejected("grant targets a different runtime or policy")
        if context["request_sha256"] != hashlib.sha256(raw_request).hexdigest():
            raise AdmissionRejected("grant payload binding mismatch")
        if context["model"] != request["model"]:
            raise AdmissionRejected("grant model mismatch")
        deadline = self._mono() + self._time_budget(context)
        frozen_request, frozen_context = _freeze(request), _freeze(context)
        expected_rows = len(frozen_request["questions"])
        if expected_rows > min(self._max_rows, context["max_question_rows"]):
            raise RequestTooLarge("question count exceeds admission limits")
        try:
            prepared = self._prepare(frozen_request)
        except InvalidRequest:
            raise
        except ValueError:
            raise InvalidRequest("request preparation rejected input") from None
        except Exception:
            raise PreparationFailed("request preparation failed") from None
        if not isinstance(prepared, PreparedInput):
            raise PreparationFailed("invalid preparation result")
        rows, tokens = prepared.question_rows, prepared.encoded_tokens
        if type(rows) is not int or type(tokens) is not int or rows <= 0 or tokens <= 0:
            raise PreparationFailed("invalid prepared request accounting")
        if rows != expected_rows:
            raise PreparationFailed("prepared row count differs from question count")
        if (rows > min(self._max_rows, context["max_question_rows"])
                or tokens > min(self._max_tokens, context["max_encoded_tokens"])):
            raise RequestTooLarge("prepared request exceeds admission limits")
        self._time_budget(context)
        if self._mono() >= deadline:
            raise AdmissionRejected("grant expired during preparation")
        try:
            accepted = self._consume(frozen_context, rows, tokens)
        except Exception:
            raise AdmissionRejected("reservation consumption failed") from None
        if accepted is not True:
            raise AdmissionRejected("reservation rejected or already consumed")
        self._time_budget(context)
        if self._mono() >= deadline:
            raise AdmissionRejected("grant expired during reservation consumption")
        return AdmittedRequest(frozen_context, frozen_request,
                               PreparedInput(rows, tokens, _freeze(prepared.payload)), deadline)

    def submit(self, worker, raw_request: bytes, envelope: bytes):
        """Authenticate and consume once per submission, then queue trusted work.

        Queue rejection can occur after reservation consumption. The platform
        must reconcile/release that reservation; never retry by bypassing the
        ledger or treating local request IDs as durable idempotency records.
        """
        admitted = self.admit(raw_request, envelope)
        return worker.submit(admitted.context["tenant_id"],
                             admitted.context["execution_attempt_id"],
                             admitted.prepared.payload,
                             cost=admitted.prepared.encoded_tokens,
                             deadline=admitted.deadline)
