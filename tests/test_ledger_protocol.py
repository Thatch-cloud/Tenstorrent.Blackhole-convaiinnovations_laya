import copy
import json
from pathlib import Path
from types import MappingProxyType

import pytest
from jsonschema import Draft202012Validator

from laya_tt.contracts import load_schema
from laya_tt.ledger_protocol import ConsumptionEncodingError, encode_consumption


CONTEXT = dict(schema_version="1", request_id="request", execution_attempt_id="attempt",
    usage_event_id="usage", tenant_id="tenant", key_id="key", runtime_generation="generation",
    policy_revision="policy", reservation_id="reservation", model="laya-english",
    checkpoint_revision="a" * 40, issued_at_unix_ms=1000, deadline_unix_ms=3000,
    max_question_rows=2, max_encoded_tokens=20, request_sha256="b" * 64)
RUNTIME = dict(model="laya-english", checkpoint_revision="a" * 40,
    runtime_revision="fixture-build", runtime_generation="generation", backend="cpu-reference")
FIXTURE = Path(__file__).resolve().parents[1] / "schemas/fixtures/reservation-consumption.json"


def encode(context=None, runtime=None, rows=2, tokens=17):
    return encode_consumption(CONTEXT if context is None else context,
                              RUNTIME if runtime is None else runtime,
                              question_rows=rows, encoded_tokens=tokens)


def test_producer_matches_portable_fixture_without_mutating_inputs():
    context, runtime = copy.deepcopy(CONTEXT), copy.deepcopy(RUNTIME)
    payload = encode(MappingProxyType(context), MappingProxyType(runtime))
    assert payload == FIXTURE.read_bytes().rstrip(b"\n")
    schema = load_schema("reservation-consumption")
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(json.loads(payload))
    assert context == CONTEXT and runtime == RUNTIME
    binding = json.loads(payload)["binding"]
    assert binding["max_encoded_tokens"] == 20
    assert json.loads(payload)["prepared_encoded_tokens"] == 17
    assert "issued_at_unix_ms" not in binding and "host_id" not in binding


@pytest.mark.parametrize("change", [
    {"runtime_generation": "other"}, {"checkpoint_revision": "c" * 40},
    {"model": "other"}, {"backend": "unverified"}, {"runtime_revision": ""},
    {"host_id": "untrusted-host"}, {"prompt": "private input"},
])
def test_runtime_identity_cannot_be_replaced_or_extended(change):
    with pytest.raises(ConsumptionEncodingError, match="^invalid reservation consumption metadata$"):
        encode(runtime=dict(RUNTIME, **change))


@pytest.mark.parametrize("change", [
    {"tenant_id": ""}, {"request_sha256": "invalid"}, {"prompt": "private input"},
    {"issued_at_unix_ms": 3000}, {"issued_at_unix_ms": 1000.0},
    {"deadline_unix_ms": True}, {"deadline_unix_ms": 2**63},
    {"max_question_rows": 2.0}, {"max_encoded_tokens": 20.0},
])
def test_invalid_claims_do_not_escape_in_errors(change):
    with pytest.raises(ConsumptionEncodingError, match="^invalid reservation consumption metadata$"):
        encode(context=dict(CONTEXT, **change))


@pytest.mark.parametrize("rows,tokens", [(0,17),(3,17),(2,0),(2,21),(True,17),(2,17.0)])
def test_prepared_work_must_be_integer_and_fit_the_exact_reservation(rows, tokens):
    with pytest.raises(ConsumptionEncodingError):
        encode(rows=rows, tokens=tokens)


def test_closed_wire_schema_excludes_host_credentials_and_customer_content():
    validator = Draft202012Validator(load_schema("reservation-consumption"))
    payload = json.loads(encode())
    for field in ("host_id", "node_token", "grant", "state", "questions"):
        assert not validator.is_valid(dict(payload, **{field: "private"}))
    changed = copy.deepcopy(payload)
    changed["binding"]["runtime"]["backend"] = "unverified"
    assert not validator.is_valid(changed)
