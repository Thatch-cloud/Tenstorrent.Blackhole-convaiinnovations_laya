import importlib.util
import json
from pathlib import Path
import socket
from unittest.mock import patch
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/compiler_spike.py"
SPEC = importlib.util.spec_from_file_location("compiler_spike", PATH)
spike = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(spike)

def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return spike.sha256_file(path)

def fixture(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    cases_path = tmp_path / "cases.json"
    reference_path = tmp_path / "reference.json"
    manifest = {"schema_version": 1}
    definitions = [{"id":"sample","questions":{}}]
    (tmp_path / "sample").mkdir()
    answer_hash = write_json(tmp_path / "sample/answers.json", [])
    tensors = {}
    for name in spike.INPUTS + spike.OUTPUTS:
        path = tmp_path / "sample" / (name + ".npy")
        path.write_bytes(b"hash-verified test fixture, not a tensor")
        tensors[name] = {"file":path.name,"sha256":spike.sha256_file(path)}
    report = {"schema_version":1,"manifest":manifest,
              "manifest_sha256":write_json(manifest_path, manifest),
              "cases_sha256":write_json(cases_path, {"cases":definitions}),
              "loaded_state_integrity":{"verified":True},
              "runtime":{"device":"cpu","weight_dtype":"torch.float32","autocast":False},
              "cases":[{"id":"sample","input":definitions[0],
                        "answers":{"file":"sample/answers.json","sha256":answer_hash},
                        "forward_calls":[tensors]}]}
    digest = write_json(reference_path, report)
    return reference_path, digest, manifest_path, cases_path

def test_validates_every_artifact_before_importing_model(tmp_path):
    args = fixture(tmp_path)
    report, _ = spike.read_reference(*args)
    assert len(report["cases"]) == 1
    (tmp_path / "sample/act_logits.npy").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        spike.read_reference(*args)

def test_rejects_index_tampering(tmp_path):
    args = fixture(tmp_path)
    args[0].write_text("{}")
    with pytest.raises(ValueError, match="index hash"):
        spike.read_reference(*args)

def test_rejects_path_escape(tmp_path):
    outside = tmp_path / "secret"
    outside.write_text("test")
    base = tmp_path / "inside"
    base.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        spike.checked_file(base, "../secret", spike.sha256_file(outside))

def lease(tmp_path, monkeypatch):
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "0")
    value = {"schema_version":1,"verified":True,"exclusive":True,
             "lease_id":"test-lease","device_serial":"physical-serial","device_id":"0",
             "host":socket.gethostname(),"expires_at":"2099-01-01T00:00:00Z"}
    path = tmp_path / "lease.json"
    return path, value

@pytest.mark.parametrize("field,value", [("verified",False),("exclusive",False),("device_serial",None),("host","other-host"),("expires_at","2000-01-01T00:00:00Z")])
def test_rejects_invalid_lease(tmp_path, monkeypatch, field, value):
    path, payload = lease(tmp_path, monkeypatch)
    payload[field] = value
    digest = write_json(path, payload)
    with pytest.raises(spike.Blocked):
        spike.read_lease(path, digest)

def test_requires_exactly_one_matching_visible_device(tmp_path, monkeypatch):
    path, payload = lease(tmp_path, monkeypatch)
    digest = write_json(path, payload)
    assert spike.read_lease(path, digest)["device_id"] == "0"
    monkeypatch.setenv("TT_VISIBLE_DEVICES", "0,1")
    with pytest.raises(spike.Blocked):
        spike.read_lease(path, digest)

def test_preflight_blocks_without_hardware_or_compiler():
    with patch.object(spike.platform, "system", return_value="Windows"), patch.object(spike.importlib.util, "find_spec", return_value=None):
        result = spike.preflight()
    assert result["status"] == "BLOCKED"
    assert result["physical_acceptance"] is False
    assert result["graph_executed"] is False
    assert len(result["reasons"]) == 3

def test_preflight_cli_writes_blocked_exit_two(tmp_path):
    reference, digest, manifest, cases = fixture(tmp_path)
    output = tmp_path / "report"
    with patch.object(spike.platform, "system", return_value="Windows"), patch.object(spike.importlib.util, "find_spec", return_value=None):
        code = spike.main(["--mode","preflight","--reference",str(reference),"--reference-sha256",digest,
                           "--manifest",str(manifest),"--cases",str(cases),"--output",str(output)])
    assert code == 2
    result = json.loads((output / "report.json").read_text())
    assert result["status"] == "BLOCKED"
    assert result["physical_acceptance"] is False

def test_numeric_comparison_detects_change():
    pytest.importorskip("numpy")
    assert spike.compare_arrays([1.,2.], [1.,2.], 0, 0)["passed"]
    assert not spike.compare_arrays([1.,2.1], [1.,2.], 0.001, 0.001)["passed"]
    with pytest.raises(ValueError, match="Non-finite"):
        spike.compare_arrays([float("nan")], [1.], 0.001, 0.001)


def test_unverified_reference_is_rejected(tmp_path):
    reference, _, manifest, cases = fixture(tmp_path)
    report = json.loads(reference.read_text())
    del report["loaded_state_integrity"]
    digest = write_json(reference, report)
    with pytest.raises(ValueError, match="verified loaded checkpoint"):
        spike.read_reference(reference, digest, manifest, cases)


def fake_distribution(commit=None, entry=True):
    from types import SimpleNamespace
    return SimpleNamespace(version="1.0.test", metadata={"Summary":"commit=" + (commit or spike.TT_XLA_AUDITED_COMMIT) + ", built-date=2026-09-25"},
                           entry_points=[SimpleNamespace(group="torch_xla.plugins", name="tt", value="torch_plugin_tt:TTPlugin")] if entry else [])

def test_compiler_requires_real_plugin_modules_not_only_torch_xla():
    with patch.object(spike.importlib.util, "find_spec", side_effect=lambda name: object() if name == "torch_xla" else None):
        with pytest.raises(spike.Blocked, match="Missing TT compiler modules"):
            spike.compiler_installation()

@pytest.mark.parametrize("commit,entry", [("a"*40, True), (None, False)])
def test_compiler_rejects_unpinned_or_unregistered_plugin(commit, entry):
    with patch.object(spike.importlib.util, "find_spec", return_value=object()), patch.object(spike.importlib.metadata, "distribution", return_value=fake_distribution(commit, entry)):
        with pytest.raises(spike.Blocked):
            spike.compiler_installation()

def test_compiler_accepts_pinned_registration_metadata():
    with patch.object(spike.importlib.util, "find_spec", return_value=object()), patch.object(spike.importlib.metadata, "distribution", return_value=fake_distribution()):
        assert spike.compiler_installation() == {"version":"1.0.test", "commit":spike.TT_XLA_AUDITED_COMMIT}

def visibility_fixture(tmp_path, monkeypatch):
    bdf = "0000:f4:00.0"
    monkeypatch.setenv("TT_VISIBLE_DEVICES", bdf)
    proof = {"verified":True, "initialization_isolated":True, "other_devices_untouched":True,
             "host":socket.gethostname(), "device_serial":"serial", "device_bdf":bdf,
             "tt_xla_commit":spike.TT_XLA_AUDITED_COMMIT, "plugin_version":"1.0.test"}
    path = tmp_path / "visibility.json"
    digest = write_json(path, proof)
    allocation = {"host":socket.gethostname(), "device_serial":"serial", "device_id":bdf,
                  "visibility_proof":{"file":path.name, "sha256":digest}}
    return proof, path, allocation, {"commit":spike.TT_XLA_AUDITED_COMMIT, "version":"1.0.test"}

def test_visibility_proof_accepts_matching_pin_and_board(tmp_path, monkeypatch):
    _, _, allocation, compiler = visibility_fixture(tmp_path, monkeypatch)
    assert spike.verify_visibility_proof(tmp_path / "lease.json", allocation, compiler)["device_bdf"] == "0000:f4:00.0"

@pytest.mark.parametrize("field,value", [("verified",False),("initialization_isolated",False),
                                        ("other_devices_untouched",False),("host","other"),
                                        ("device_serial","other"),("plugin_version","different"),
                                        ("tt_xla_commit","a"*40),("device_bdf","0")])
def test_visibility_proof_rejects_unverified_or_wrong_identity(tmp_path, monkeypatch, field, value):
    proof, path, allocation, compiler = visibility_fixture(tmp_path, monkeypatch)
    proof[field] = value
    allocation["visibility_proof"]["sha256"] = write_json(path, proof)
    with pytest.raises(spike.Blocked):
        spike.verify_visibility_proof(tmp_path / "lease.json", allocation, compiler)

def test_visibility_proof_rejects_tampering(tmp_path, monkeypatch):
    proof, path, allocation, compiler = visibility_fixture(tmp_path, monkeypatch)
    proof["verified"] = False
    write_json(path, proof)
    with pytest.raises(ValueError, match="hash mismatch"):
        spike.verify_visibility_proof(tmp_path / "lease.json", allocation, compiler)

def test_missing_visibility_proof_blocks(tmp_path):
    with pytest.raises(spike.Blocked, match="visibility proof"):
        spike.verify_visibility_proof(tmp_path / "lease.json", {}, {})

@pytest.mark.parametrize("value", ["", "/tmp/saved.ttsys"])
def test_hardware_rejects_compile_only_environment(monkeypatch, value):
    monkeypatch.setenv("TT_COMPILE_ONLY_SYSTEM_DESC", value)
    with pytest.raises(spike.Blocked, match="not inference"):
        spike.reject_compile_only_for_hardware()

def test_select_compile_call_requires_real_forward(tmp_path):
    reference, digest, manifest, cases = fixture(tmp_path)
    report, _ = spike.read_reference(reference,digest,manifest,cases)
    case, call = spike.select_compile_call(report, "sample", 0)
    assert case["id"] == "sample"
    assert set(call) == set(spike.INPUTS + spike.OUTPUTS)
    for name, index in [("absent",0),("sample",-1),("sample",1)]:
        with pytest.raises(ValueError):
            spike.select_compile_call(report,name,index)

def test_compile_artifacts_must_be_nonempty_and_are_hashed(tmp_path):
    prefix = tmp_path / "model"
    for suffix in (".ttnn","_ttnn.mlir","_ttir.mlir"):
        Path(str(prefix)+suffix).write_bytes(b"fixture")
    result = spike.compile_artifact_inventory(tmp_path,prefix)
    assert len(result) == 3
    assert all(item["sha256"] == spike.sha256_file(tmp_path/item["file"]) for item in result.values())
    Path(str(prefix)+".ttnn").write_bytes(b"")
    with pytest.raises(ValueError,match="Missing or empty"):
        spike.compile_artifact_inventory(tmp_path,prefix)

def test_compile_only_blocks_exposed_hardware(monkeypatch):
    monkeypatch.setattr(spike.platform,"system",lambda:"Linux")
    with patch.object(spike.Path,"glob",return_value=[Path("/dev/tenstorrent/0")]):
        with pytest.raises(spike.Blocked,match="without exposed"):
            spike.compile_only_preflight("descriptor",spike.P150_DESCRIPTOR_SHA256)

def test_compile_only_descriptor_and_build_guard(tmp_path,monkeypatch):
    path=tmp_path/"p150.ttsys"
    path.write_bytes(b"unit-test descriptor, not runtime evidence")
    digest=spike.sha256_file(path)
    monkeypatch.setattr(spike,"P150_DESCRIPTOR_SHA256",digest)
    monkeypatch.setattr(spike.platform,"system",lambda:"Linux")
    monkeypatch.delenv("TT_VISIBLE_DEVICES",raising=False)
    monkeypatch.delenv("TT_COMPILE_ONLY_SYSTEM_DESC",raising=False)
    with patch.object(spike.Path,"glob",return_value=[]), patch.object(spike,"compiler_installation",return_value={"commit":spike.OFFLINE_NIGHTLY_COMMIT,"version":spike.OFFLINE_NIGHTLY_VERSION}) as compiler:
        result=spike.compile_only_preflight(path,digest,spike.OFFLINE_NIGHTLY_COMMIT)
        compiler.assert_called_once_with(spike.OFFLINE_NIGHTLY_COMMIT)
        assert result["system_descriptor"]["sha256"]==digest
        assert not result["device_nodes_exposed"]
        with pytest.raises(ValueError,match="pinned SHA256"):
            spike.compile_only_preflight(path,"0"*64)
        path.write_bytes(b"tampered")
        with pytest.raises(ValueError,match="hash mismatch"):
            spike.compile_only_preflight(path,digest)

def test_nightly_requires_explicit_revision_and_version():
    distribution=fake_distribution(spike.OFFLINE_NIGHTLY_COMMIT)
    distribution.version=spike.OFFLINE_NIGHTLY_VERSION
    with patch.object(spike.importlib.util,"find_spec",return_value=object()), patch.object(spike.importlib.metadata,"distribution",return_value=distribution):
        with pytest.raises(spike.Blocked,match="pinned revision"):
            spike.compiler_installation()
        assert spike.compiler_installation(spike.OFFLINE_NIGHTLY_COMMIT)["version"]==spike.OFFLINE_NIGHTLY_VERSION
        distribution.version="different"
        with pytest.raises(spike.Blocked,match="exact audited"):
            spike.compiler_installation(spike.OFFLINE_NIGHTLY_COMMIT)

def test_compile_only_cli_does_not_report_execution(tmp_path,monkeypatch):
    reference,digest,manifest,cases=fixture(tmp_path)
    output=tmp_path/"compile-report"
    monkeypatch.delenv("TT_COMPILE_ONLY_SYSTEM_DESC",raising=False)
    def compile_stub(*args):
        return {"compilation_status":"COMPILED","artifacts":{"fixture":{}},"numerical_comparison_performed":False,"physical_acceptance":False}
    with patch.object(spike,"compile_only_preflight",return_value={"device_nodes_exposed":False}), patch.object(spike,"validate_manifest",return_value=(tmp_path,tmp_path)), patch.object(spike,"run_compile_only",side_effect=compile_stub), patch.object(spike,"run_graphs",side_effect=AssertionError("Inference path forbidden")):
        code=spike.main(["--mode","tt-compile-only","--reference",str(reference),"--reference-sha256",digest,"--manifest",str(manifest),"--cases",str(cases),"--output",str(output),"--case-id","sample"])
    report=json.loads((output/"report.json").read_text())
    assert code==0
    assert report["status"]=="COMPILED"
    assert report["compilation_status"]=="COMPILED"
    assert "graph_executed" not in report
    assert "device_execution" not in report
    assert not report["physical_acceptance"]
