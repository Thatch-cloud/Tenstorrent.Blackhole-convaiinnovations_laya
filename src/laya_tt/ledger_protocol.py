"""Closed consumption encoding; authentication and delivery remain platform-owned."""
from collections.abc import Mapping
import json

from jsonschema import Draft202012Validator

from .contracts import load_schema


class ConsumptionEncodingError(ValueError):
    """Fixed error without claim, customer input, or credential diagnostics."""


_CONTEXT = Draft202012Validator(load_schema("admission-context"))
_CONSUMPTION = Draft202012Validator(load_schema("reservation-consumption"))
_IDENTITIES = ("tenant_id", "key_id", "request_id", "execution_attempt_id",
               "reservation_id", "usage_event_id", "request_sha256", "policy_revision",
               "max_question_rows", "max_encoded_tokens")


def encode_consumption(context: Mapping, runtime: Mapping, *,
                       question_rows: int, encoded_tokens: int) -> bytes:
    """Bind verified claims and independently observed loaded-runtime metadata.

    Call only after grant authentication and trusted preparation. ``runtime`` is
    supplied by bootstrap/readback, never by public request JSON. This function
    neither authenticates a grant, sends a request, nor permits execution. The
    caller must obtain the authenticated receiver's committed success exactly
    once; missing/uncertain acknowledgment requires reconciliation, not retry.
    """
    try:
        if not isinstance(context, Mapping) or not isinstance(runtime, Mapping):
            raise ValueError()
        # Own the snapshot; these inputs may be read-only mapping proxies.
        claims = json.loads(json.dumps(dict(context), allow_nan=False))
        observed = json.loads(json.dumps(dict(runtime), allow_nan=False))
        _CONTEXT.validate(claims)
        for key in ("issued_at_unix_ms", "deadline_unix_ms", "max_question_rows",
                    "max_encoded_tokens"):
            if type(claims[key]) is not int:
                raise ValueError()
        if not 0 < claims["issued_at_unix_ms"] < claims["deadline_unix_ms"] <= 2**63 - 1:
            raise ValueError()
        if any(observed[key] != claims[key] for key in
               ("model", "checkpoint_revision", "runtime_generation")):
            raise ValueError()
        if (type(question_rows) is not int or type(encoded_tokens) is not int
                or not 1 <= question_rows <= claims["max_question_rows"]
                or not 1 <= encoded_tokens <= claims["max_encoded_tokens"]):
            raise ValueError()
        binding = {key: claims[key] for key in _IDENTITIES}
        binding["runtime"] = observed
        payload = {"binding": binding, "deadline_unix_ms": claims["deadline_unix_ms"],
                   "prepared_question_rows": question_rows,
                   "prepared_encoded_tokens": encoded_tokens}
        _CONSUMPTION.validate(payload)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(encoded) > 16 * 1024:
            raise ValueError()
        return encoded
    except Exception:
        raise ConsumptionEncodingError("invalid reservation consumption metadata") from None
