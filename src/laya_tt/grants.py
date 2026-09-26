"""Pinned-key verification of the version-one decision JWS profile."""
import base64
import json
import re
import time

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from jsonschema import Draft202012Validator

from .contracts import load_schema
from .ledger_protocol import encode_consumption


class InvalidGrant(ValueError):
    """Fixed diagnostics; no token or claim contents."""


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError()
        value[key] = item
    return value


def _json(raw):
    def invalid(_value):
        raise ValueError()
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=invalid)


def _decode(part):
    if not re.fullmatch(rb"[A-Za-z0-9_-]+", part):
        raise ValueError()
    value = base64.urlsafe_b64decode(part + b"=" * (-len(part) % 4))
    if base64.urlsafe_b64encode(value).rstrip(b"=") != part:
        raise ValueError()
    return value


class DecisionGrantVerifier:
    """Explicit trusted assignment; no key discovery, credentials or signing.

    The host supplies a pinned key set for one issuer, stable host identity,
    loaded runtime and policy. Replace the verifier to rotate/revoke keys.
    A signature authenticates claims, not consumption or device ownership.
    """
    def __init__(self, *, public_keys, issuer, host_id, runtime, policy_revision,
                 wall_time=time.time):
        try:
            if any(not isinstance(value, str) or not 1 <= len(value) <= 128
                   for value in (issuer, host_id, policy_revision)) or not callable(wall_time):
                raise ValueError()
            if not 1 <= len(public_keys) <= 16:
                raise ValueError()
            keys = {}
            for kid, raw in public_keys.items():
                if (not isinstance(kid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", kid)
                        or type(raw) is not bytes or len(raw) != 32):
                    raise ValueError()
                keys[kid] = Ed25519PublicKey.from_public_bytes(raw)
            observed = _json(json.dumps(dict(runtime), allow_nan=False).encode())
            schema = load_schema("execution-receipt")
            Draft202012Validator({"$ref": "#/$defs/runtime", "$defs": schema["$defs"]}).validate(observed)
        except Exception:
            raise ValueError("invalid decision grant trust assignment") from None
        self._keys = keys
        self._issuer = issuer
        self._host = host_id
        self._runtime = observed
        self._policy = policy_revision
        self._clock = wall_time

    def __call__(self, envelope):
        try:
            if type(envelope) is not bytes or not 1 <= len(envelope) <= 12288:
                raise ValueError()
            protected, payload, signature = envelope.split(b".")
            header = _json(_decode(protected))
            if (type(header) is not dict or set(header) != {"alg", "typ", "kid"}
                    or header["alg"] != "Ed25519" or header["typ"] != "thatch-decision-grant+jws"
                    or type(header["kid"]) is not str):
                raise ValueError()
            raw_signature = _decode(signature)
            if len(raw_signature) != 64:
                raise ValueError()
            self._keys[header["kid"]].verify(raw_signature, protected + b"." + payload)
            value = _json(_decode(payload))
            if (type(value) is not dict
                    or set(value) != {"schema_version", "issuer", "host_id", "runtime", "context"}
                    or value["schema_version"] != "1" or value["issuer"] != self._issuer
                    or value["host_id"] != self._host or value["runtime"] != self._runtime):
                raise ValueError()
            context = value["context"]
            # Reuse the closed claims/runtime and strict integer validation.
            encode_consumption(context, self._runtime, question_rows=1, encoded_tokens=1)
            if context["policy_revision"] != self._policy:
                raise ValueError()
            now_ms = self._clock() * 1000
            if not context["issued_at_unix_ms"] <= now_ms < context["deadline_unix_ms"]:
                raise ValueError()
            return context
        except Exception:
            raise InvalidGrant("decision grant verification failed") from None
