"""Offline-only FlatBuffers schema migration; never queries a device."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

OLD_MLIR = "464a5f341908d5a3f790e697be5ffa6fd3fb6f32"
NEW_MLIR = "e2b21a721955b2b68540d9e6879a215ce3dcfae5"
FLATBUFFERS = "fb9afbafc7dfe226b9db54d4923bfb8839635274"
ORIGINAL_SHA = "655fabe5445624b0c9a6a312b90fb5e9a30a78614e43d331e9d622561e09b5ac"
OLD_SCHEMA_HASH = "7f252e8a9f2f1850b406e7c18440842ccf8d0c51c47b8747c297e5b6bb8a966f"
NEW_SCHEMA_HASH = "92f6a8a87b72abd510d53425c6cd90ab2e8ba161928ee94c8bdd6680f51d9cdb"
SHARED_HASHES = {
    "system_desc.fbs": "b8198760bac728c8437bc9d8dad644722ad4163c17ea57d383eb47dc7a71648b",
    "version.fbs": "e8d00be3a6d4dc46cc7fa64d92f6cfcb88f6c7decddd889d039367c2076d7d15",
}
SCHEMA_HASHES = {
    "old": dict(SHARED_HASHES, **{"types.fbs": "19d0509cb50ac07eb04af6b504601d71f111871addd0ca5bc7e50b20fe4af8e3"}),
    "new": dict(SHARED_HASHES, **{"types.fbs": "6c3fa6c6277da672b6b3514f1c89ab63e1123fdcd3f23c0f7f76892309afff90"}),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_digest(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {Path(path).name}: {actual}")


def run(argv, cwd=None):
    result = subprocess.run([str(x) for x in argv], cwd=cwd, check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


def validate_original(document):
    if document.get("schema_hash") != OLD_SCHEMA_HASH:
        raise ValueError("Unexpected original schema hash")
    if document.get("ttmlir_git_hash") != OLD_MLIR:
        raise ValueError("Unexpected original descriptor producer")
    system = document.get("system_desc")
    if not isinstance(system, dict) or not system.get("chip_descs"):
        raise ValueError("Missing hardware description")
    if len(system.get("chip_desc_indices", [])) != 1:
        raise ValueError("Expected one-chip generic descriptor")


def assert_preserved(original, migrated):
    # Compare the entire decoded root, including every nested topology field.
    expected = dict(original)
    expected["schema_hash"] = NEW_SCHEMA_HASH
    if migrated != expected:
        raise ValueError("Migration changed data beyond the allowed schema_hash metadata")


def prepare_schemas(bundle, work):
    roots = {}
    for revision, hashes in SCHEMA_HASHES.items():
        root = work / revision / "include"
        common = root / "ttmlir/Target/Common"
        common.mkdir(parents=True)
        for name, expected in hashes.items():
            source = bundle / revision / name
            require_digest(source, expected)
            (common / name).write_bytes(source.read_bytes())
        roots[revision] = root
    return roots


def generate_schema_hash(flatc, root, destination):
    destination.mkdir()
    common = root / "ttmlir/Target/Common"
    # Exact options/order from pinned MLIR cmake/modules/BuildFlatbuffers.cmake.
    run([flatc, "-I", root, "--bfbs-gen-embed", "--cpp", "--cpp-std", "c++17",
         "--scoped-enums", "--warnings-as-errors", "--keep-prefix",
         "--gen-name-strings", "--gen-object-api", "-o", destination,
         "system_desc.fbs"], cwd=common)
    # sha256-include-gen.py hashes UTF-8 text of this generated header.
    header = destination / "system_desc_bfbs_generated.h"
    actual = hashlib.sha256(header.read_text(encoding="utf-8").encode()).hexdigest()
    if actual != NEW_SCHEMA_HASH:
        raise ValueError(f"Generated BFBS schema hash differs from installed compiler: {actual}")
    return actual


def decode(flatc, root, binary, destination):
    destination.mkdir()
    run([flatc, "--json", "--strict-json", "--defaults-json", "--size-prefixed",
         "-I", root, "-o", destination,
         root / "ttmlir/Target/Common/system_desc.fbs", "--", binary])
    return json.loads((destination / (binary.stem + ".json")).read_text())


def migrate(args):
    original = Path(args.input).resolve()
    flatc = Path(args.flatc).resolve()
    flatbuffers_source = Path(args.flatbuffers_source).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("Output directory must not exist; original artifacts are immutable")
    if run(["git", "-C", flatbuffers_source, "rev-parse", "HEAD"]) != FLATBUFFERS:
        raise ValueError("FlatBuffers source revision mismatch")
    if run(["git", "-C", flatbuffers_source, "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("FlatBuffers source has tracked modifications")
    require_digest(original, ORIGINAL_SHA)
    flatc_hash = digest(flatc)
    flatc_version = run([flatc, "--version"])
    with tempfile.TemporaryDirectory(prefix="laya-descriptor-migration-") as temporary:
        work = Path(temporary)
        roots = prepare_schemas(Path(args.schemas).resolve(), work)
        schema_hash = generate_schema_hash(flatc, roots["new"], work / "generated")
        before = decode(flatc, roots["old"], original, work / "original-json")
        validate_original(before)
        migrated = dict(before)
        # Preserve original producer/version and all hardware data. The sidecar
        # identifies this conversion; it does not claim a new hardware capture.
        migrated["schema_hash"] = schema_hash
        encoding_input = work / "p150-migrated-nightly.json"
        encoding_input.write_text(json.dumps(migrated, sort_keys=True, indent=2), encoding="utf-8")
        encoded = work / "encoded"
        encoded.mkdir()
        run([flatc, "--binary", "--size-prefixed", "-I", roots["new"], "-o", encoded,
             roots["new"] / "ttmlir/Target/Common/system_desc.fbs", encoding_input])
        binary = encoded / "p150-migrated-nightly.ttsys"
        after = decode(flatc, roots["new"], binary, work / "migrated-json")
        assert_preserved(before, after)
        # Validate backward-readable topology too, rather than trusting one decoder.
        cross = decode(flatc, roots["old"], binary, work / "old-schema-roundtrip")
        assert_preserved(before, cross)
        require_digest(original, ORIGINAL_SHA)
        if digest(flatc) != flatc_hash:
            raise ValueError("flatc changed during migration")
        provenance = {
            "schema_version": 1, "kind": "migrated_generic_system_descriptor",
            "offline_only": True, "current_card_inventory": False,
            "physical_acceptance": False, "hardware_queried": False,
            "source_tt_xla_commit": "873c53c4ff84bd91112a1f43e788cfed75aaa914",
            "original_descriptor_sha256": ORIGINAL_SHA,
            "original_mlir_commit": OLD_MLIR, "target_mlir_commit": NEW_MLIR,
            "original_schema_hash": OLD_SCHEMA_HASH, "target_schema_hash": schema_hash,
            "flatbuffers_commit": FLATBUFFERS, "flatc_version": flatc_version,
            "flatc_sha256": flatc_hash, "schema_file_sha256": SCHEMA_HASHES,
            "descriptor": {"file": binary.name, "sha256": digest(binary)},
            "allowed_metadata_changes": ["schema_hash"],
            "original_producer_metadata_preserved": True,
            "complete_root_roundtrip_equal_except_allowed_metadata": True,
            "old_schema_roundtrip_equal_except_allowed_metadata": True,
            "note": "Offline schema migration of upstream generic P150 fixture; not a capture of the current card",
        }
        output.mkdir(parents=True)
        (output / binary.name).write_bytes(binary.read_bytes())
        for name, document in (("original-decoded.json", before), ("migrated-decoded.json", after),
                               ("provenance.json", provenance)):
            (output / name).write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--flatc", required=True)
    parser.add_argument("--flatbuffers-source", required=True)
    parser.add_argument("--schemas", default=str(Path(__file__).resolve().parents[1] / "configs/descriptor-schemas"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = migrate(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
