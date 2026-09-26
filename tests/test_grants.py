import base64
import json

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from laya_tt.grants import DecisionGrantVerifier, InvalidGrant
from test_ledger_client import RUNTIME, claims

# Public test seed only. No production signer exists in this runtime package.
KEY = Ed25519PrivateKey.from_private_bytes(bytes([42]) * 32)


def b64(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=")


def context():
    return dict(claims(), issued_at_unix_ms=1000, deadline_unix_ms=3000)


def payload():
    return dict(schema_version="1", issuer="authority", host_id="host",
                runtime=dict(RUNTIME), context=context())


def sign(value=None, header=None, key=KEY, raw_payload=None, raw_header=None):
    h = b64(raw_header if raw_header is not None else json.dumps(header or
            dict(alg="Ed25519", typ="thatch-decision-grant+jws", kid="test-key")).encode())
    p = b64(raw_payload if raw_payload is not None else json.dumps(value or payload()).encode())
    message = h + b"." + p
    return message + b"." + b64(key.sign(message))


def verifier(**overrides):
    options = dict(public_keys={"test-key": KEY.public_key().public_bytes_raw()}, issuer="authority",
                   host_id="host", runtime=RUNTIME, policy_revision="policy", wall_time=lambda: 2)
    options.update(overrides)
    return DecisionGrantVerifier(**options)


def test_verified_owned_claims_and_key_snapshot():
    keys = {"test-key": KEY.public_key().public_bytes_raw()}
    runtime = dict(RUNTIME)
    verify = verifier(public_keys=keys, runtime=runtime)
    keys.clear()
    runtime["runtime_generation"] = "mutated"
    value = verify(sign())
    assert value == context()
    value["tenant_id"] = "mutated"
    assert verify(sign())["tenant_id"] == "tenant"


@pytest.mark.parametrize("field", ["issuer", "host_id", "schema_version"])
def test_signed_wrong_assignment(field):
    value = payload()
    value[field] = "wrong"
    with pytest.raises(InvalidGrant):
        verifier()(sign(value))


@pytest.mark.parametrize("field", list(RUNTIME))
def test_signed_wrong_runtime(field):
    value = payload()
    value["runtime"][field] = "wrong"
    with pytest.raises(InvalidGrant):
        verifier()(sign(value))


@pytest.mark.parametrize("change", [
    {"policy_revision": "wrong"}, {"issued_at_unix_ms": 2001}, {"deadline_unix_ms": 2000},
    {"issued_at_unix_ms": True}, {"deadline_unix_ms": 3000.0}, {"max_encoded_tokens": False},
    {"max_question_rows": 65}, {"extra": "claim"}, {"checkpoint_revision": "wrong"},
])
def test_signed_invalid_claims(change):
    value = payload()
    value["context"].update(change)
    with pytest.raises(InvalidGrant, match="^decision grant verification failed$"):
        verifier()(sign(value))


@pytest.mark.parametrize("change", [{"alg": "EdDSA"}, {"alg": "none"}, {"alg": "HS256"}, {"typ": "JWT"},
    {"kid": "unknown"}, {"jku": "https://untrusted.invalid/keys"}, {"crit": []}, {"b64": False}])
def test_closed_protected_header(change):
    header = dict(alg="Ed25519", typ="thatch-decision-grant+jws", kid="test-key")
    header.update(change)
    with pytest.raises(InvalidGrant):
        verifier()(sign(header=header))


@pytest.mark.parametrize("bad", ["signature", "key", "payload", "padded", "extra-part", "empty", "oversize",
                                 "duplicate-header", "duplicate-payload", "duplicate-context", "array", "extra-field"])
def test_invalid_envelope(bad):
    token = sign()
    if bad == "signature":
        h, p, s = token.split(b".")
        token = h + b"." + p + b"." + b64(bytes(64))
    elif bad == "key":
        token = sign(key=Ed25519PrivateKey.generate())
    elif bad == "payload":
        h, p, s = token.split(b".")
        token = h + b"." + b64(json.dumps(dict(payload(), host_id="other")).encode()) + b"." + s
    elif bad == "padded":
        token += b"="
    elif bad == "extra-part":
        token += b".extra"
    elif bad == "empty":
        token = b""
    elif bad == "oversize":
        token = b"a" * 12289
    elif bad == "duplicate-header":
        token = sign(raw_header=b'{"alg":"Ed25519","alg":"Ed25519","typ":"thatch-decision-grant+jws","kid":"test-key"}')
    elif bad == "duplicate-payload":
        token = sign(raw_payload=json.dumps(payload()).replace('"host_id":', '"host_id":"host","host_id":').encode())
    elif bad == "duplicate-context":
        token = sign(raw_payload=json.dumps(payload()).replace('"tenant_id":', '"tenant_id":"tenant","tenant_id":').encode())
    elif bad == "array":
        token = sign(raw_payload=b"[]")
    else:
        token = sign(dict(payload(), customer_text="never accepted"))
    with pytest.raises(InvalidGrant):
        verifier()(token)


@pytest.mark.parametrize("keys", [{}, {"bad key": bytes(32)}, {"key": b"short"},
                                  {str(n): bytes(32) for n in range(17)}])
def test_bad_trust_assignment(keys):
    with pytest.raises(ValueError, match="invalid decision grant trust assignment"):
        verifier(public_keys=keys)


def test_explicit_key_rotation_and_revocation():
    other = Ed25519PrivateKey.generate()
    token = sign(header=dict(alg="Ed25519", typ="thatch-decision-grant+jws", kid="next"), key=other)
    verify = verifier(public_keys={"test-key": KEY.public_key().public_bytes_raw(),
                                   "next": other.public_key().public_bytes_raw()})
    assert verify(token) == context()
    assert verify(sign()) == context()
    revoked = verifier(public_keys={"next": other.public_key().public_bytes_raw()})
    with pytest.raises(InvalidGrant):
        revoked(sign())
    assert revoked(token) == context()


def test_signed_grant_admission_still_requires_payload_binding_and_consumption():
    import hashlib
    from laya_tt.admission import AdmissionAdapter, AdmissionRejected, PreparedInput
    raw = json.dumps({"model": "laya-english", "state": "fixture",
                      "questions": {"q": {"type": "noul", "instructions": "Evaluate"}}}).encode()
    value = payload()
    value["context"]["request_sha256"] = hashlib.sha256(raw).hexdigest()
    calls = []
    def consume(context, rows, tokens):
        calls.append((context, rows, tokens))
        return len(calls) == 1
    adapter = AdmissionAdapter(verify_grant=verifier(), prepare=lambda _: PreparedInput(1, 5, "fixture"),
        consume_reservation=consume, checkpoint_revision=RUNTIME["checkpoint_revision"],
        runtime_generation="generation", policy_revision="policy", wall_time=lambda: 2)
    token = sign(value)
    with pytest.raises(AdmissionRejected):
        adapter.admit(raw.replace(b"fixture", b"changed"), token)
    assert not calls
    adapter.admit(raw, token)
    assert len(calls) == 1
    with pytest.raises(AdmissionRejected):
        adapter.admit(raw, token)
    assert len(calls) == 2  # The signature never bypasses single-use consumption.


def test_portable_signed_fixture():
    from pathlib import Path
    value = json.loads((Path(__file__).parent / "fixtures" / "decision-grant-v1.json").read_text())
    verify = verifier(public_keys={"test-key": bytes.fromhex(value["public_key_hex"])})
    assert verify(value["grant"].encode("ascii")) == value["context"]
