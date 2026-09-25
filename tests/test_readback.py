import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from laya_tt.admission import AdmissionAdapter
from laya_tt.asgi import DecisionASGI
from laya_tt.cpu_backend import CpuReferenceBackend
from laya_tt.readback import BackendReadback, ReadbackUnavailable
from laya_tt.service import DecisionService, RuntimeIdentity, ServiceNotReady


def facts(**overrides):
    args = dict(backend="cpu-reference", precision="float32",
                question_types=("choice", "score", "noul"), max_question_rows=64,
                max_candidates_per_question=64, max_sequence_tokens=512,
                max_head_tokens=192, max_encoded_tokens=32768)
    args.update(overrides)
    return BackendReadback(**args)


@pytest.fixture
def make_service():
    services = []
    def make(*, provider=facts, max_rows=64, max_tokens=32768, receipt_sink=None):
        def forbidden(*args):
            pytest.fail("readback invoked admission or execution")
        admission = AdmissionAdapter(verify_grant=forbidden, prepare=forbidden,
            consume_reservation=forbidden, checkpoint_revision="a" * 40,
            runtime_generation="test-generation", policy_revision="policy",
            max_question_rows=max_rows, max_encoded_tokens=max_tokens)
        service = DecisionService(admission=admission, backend=forbidden,
            runtime=RuntimeIdentity("a" * 40, "revision", "test-generation", "cpu-reference"),
            backend_readback=provider, receipt_sink=receipt_sink)
        services.append(service)
        return service
    yield make
    for service in services:
        service.drain(timeout=2)


async def request(service, path="/internal/decision-runtime", method="GET"):
    messages = []
    async def receive():
        pytest.fail("readback should not consume a body or admission grant")
    async def send(value):
        messages.append(value)
    await DecisionASGI(service)({"type": "http", "path": path, "method": method}, receive, send)
    return messages[0]["status"], json.loads(messages[1]["body"]), messages[0]["headers"]


def test_readback_exact_neutral_shape_without_collector_clock(make_service):
    service = make_service(max_rows=3, max_tokens=10000)
    status, body, headers = asyncio.run(request(service))
    assert status == 200
    assert body == dict(contract_version=1, model="laya-english", checkpoint_revision="a" * 40,
        runtime_revision="revision", runtime_generation="test-generation", backend="cpu-reference",
        precision="float32", state="starting", question_types=["choice", "score", "noul"],
        limits=dict(max_request_bytes=2097152, max_question_rows=3,
                    max_candidates_per_question=64, max_sequence_tokens=512,
                    max_head_tokens=192, max_encoded_tokens=1536))
    assert (b"cache-control", b"no-store") in headers
    assert not service.readiness()["ready"]


def test_effective_bounds_intersect_backend_and_admission(make_service):
    service = make_service(max_rows=4, max_tokens=700, provider=lambda: facts(
        max_question_rows=2, max_candidates_per_question=8, max_sequence_tokens=256,
        max_head_tokens=128, max_encoded_tokens=500, question_types=("noul",)))
    observed = service.runtime_readback()
    assert observed["limits"]["max_encoded_tokens"] == 500
    assert observed["limits"]["max_question_rows"] == 2
    assert observed["limits"]["max_candidates_per_question"] == 8
    assert observed["question_types"] == ["noul"]


def test_lifecycle_readback_does_not_start_or_change_health_contract(make_service):
    service = make_service()
    assert asyncio.run(request(service, "/readyz"))[0] == 503
    service.start(lambda: True)  # Synthetic model fixture only.
    assert service.runtime_readback()["state"] == "ready"
    assert asyncio.run(request(service, "/readyz"))[1] == service.readiness()
    assert service.drain()
    assert service.runtime_readback()["state"] == "stopped"
    assert asyncio.run(request(service, "/healthz"))[0] == 200
    assert asyncio.run(request(service, "/readyz"))[0] == 503


def test_failed_self_test_is_observed(make_service):
    service = make_service()
    with pytest.raises(ServiceNotReady):
        service.start(lambda: False)
    assert service.runtime_readback()["state"] == "failed"


def test_draining_during_boot_is_observed_without_becoming_ready(make_service):
    service = make_service()
    entered, release = threading.Event(), threading.Event()
    errors = []
    def self_test():
        entered.set()
        assert release.wait(2)
        return True
    def start():
        try:
            service.start(self_test)
        except ServiceNotReady as exc:
            errors.append(exc)
    thread = threading.Thread(target=start)
    thread.start()
    try:
        assert entered.wait(1)
        assert not service.drain(wait=False)
        assert service.runtime_readback()["state"] == "draining"
        assert asyncio.run(request(service, "/readyz"))[0] == 503
    finally:
        release.set()
        thread.join(2)
    assert len(errors) == 1
    assert service.drain()


def test_recovery_gate_runs_before_model_test_and_failure_withdraws_startup(make_service):
    calls = []
    def check_startup():
        calls.append("reconciliation")
        raise RuntimeError("unresolved previous execution")
    sink = SimpleNamespace(check_startup=check_startup,
        **{name: lambda *args: None for name in
           ("admitted", "started", "completed", "failed", "not_started", "delivery")})
    service = make_service(receipt_sink=sink)
    with pytest.raises(RuntimeError):
        service.start(lambda: calls.append("model") or True)
    assert calls == ["reconciliation"]
    assert service.runtime_readback()["state"] == "failed"


def test_reconciled_journal_permits_explicit_model_test(make_service):
    calls = []
    sink = SimpleNamespace(check_startup=lambda: calls.append("reconciliation"),
        **{name: lambda *args: None for name in
           ("admitted", "started", "completed", "failed", "not_started", "delivery")})
    service = make_service(receipt_sink=sink)
    service.start(lambda: calls.append("model") or True)
    assert calls == ["reconciliation", "model"]
    assert service.runtime_readback()["state"] == "ready"


@pytest.mark.parametrize("provider", [None, lambda: {}, lambda: facts(backend="tt-blackhole")])
def test_missing_invalid_mismatched_backend_fails_closed(make_service, provider):
    service = make_service(provider=provider)
    status, body, _ = asyncio.run(request(service))
    assert status == 503
    assert body == {"error": {"code": "runtime_readback_unavailable"}}
    assert service.readiness()["state"] == "starting"


def test_readback_provider_failure_is_sanitized_and_not_cached(make_service):
    calls = []
    def provider():
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("secret backend detail")
        return facts()
    service = make_service(provider=provider)
    assert asyncio.run(request(service))[0] == 200
    assert asyncio.run(request(service))[1] == {"error": {"code": "runtime_readback_unavailable"}}


def test_method_is_get_only(make_service):
    status, _, headers = asyncio.run(request(make_service(), method="POST"))
    assert status == 405
    assert (b"allow", b"GET") in headers


@pytest.mark.parametrize("overrides", [{"precision": ""}, {"precision": "fp32\n"},
    {"max_head_tokens": 513}, {"max_encoded_tokens": 32769},
    {"question_types": ("choice", "choice")}, {"max_question_rows": True}])
def test_invalid_backend_facts_rejected(overrides):
    with pytest.raises(ValueError):
        facts(**overrides)


def test_cpu_readback_requires_factory_verification_and_current_precision():
    parameter = SimpleNamespace(device=SimpleNamespace(type="cpu"), dtype="float32",
                                is_floating_point=lambda: True)
    agent = SimpleNamespace(device=SimpleNamespace(type="cpu"), amp_enabled=False,
                            _fast=None, model=SimpleNamespace(parameters=lambda: [parameter], buffers=lambda: []))
    backend = CpuReferenceBackend(agent, None, SimpleNamespace(float32="float32"))
    with pytest.raises(ReadbackUnavailable):
        backend.readback()
    backend._bootstrap_verified = True  # Simulate verified factory, no real model claim.
    assert backend.readback().precision == "float32"
    parameter.dtype = "bfloat16"
    with pytest.raises(ReadbackUnavailable):
        backend.readback()
    parameter.dtype = "float32"
    buffer = SimpleNamespace(device=SimpleNamespace(type="cpu"), dtype="bfloat16",
                             is_floating_point=lambda: True)
    agent.model.buffers = lambda: [buffer]
    with pytest.raises(ReadbackUnavailable):
        backend.readback()
    buffer.dtype = "float32"
    assert backend.readback().precision == "float32"
    buffer.device.type = "cuda"
    with pytest.raises(ReadbackUnavailable):
        backend.readback()
