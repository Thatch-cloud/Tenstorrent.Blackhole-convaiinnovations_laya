import copy
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/migrate_system_descriptor.py"
SPEC = importlib.util.spec_from_file_location("migration", PATH)
migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migration)


def original():
    return {"schema_hash": migration.OLD_SCHEMA_HASH, "ttmlir_git_hash": migration.OLD_MLIR,
            "version": {"major": 0, "minor": 1, "patch": 0},
            "system_desc": {"chip_desc_indices": [0], "chip_descs": [
                {"arch": "Blackhole", "grid_size": {"x": 14, "y": 10},
                 "dram_channel_size": 4294967296}], "chip_channels": []}}


def test_only_schema_metadata_change_allowed():
    before = original()
    migration.validate_original(before)
    after = copy.deepcopy(before)
    after["schema_hash"] = migration.NEW_SCHEMA_HASH
    migration.assert_preserved(before, after)
    assert before["schema_hash"] == migration.OLD_SCHEMA_HASH


@pytest.mark.parametrize("change", ["capacity", "grid", "producer", "version", "extra", "missing"])
def test_any_other_difference_is_rejected(change):
    before = original()
    after = copy.deepcopy(before)
    after["schema_hash"] = migration.NEW_SCHEMA_HASH
    if change == "capacity":
        after["system_desc"]["chip_descs"][0]["dram_channel_size"] += 1
    elif change == "grid":
        after["system_desc"]["chip_descs"][0]["grid_size"]["x"] += 1
    elif change == "producer":
        after["ttmlir_git_hash"] = migration.NEW_MLIR
    elif change == "version":
        after["version"]["patch"] += 1
    elif change == "extra":
        after["system_desc"]["invented"] = True
    else:
        del after["system_desc"]["chip_channels"]
    with pytest.raises(ValueError, match="beyond"):
        migration.assert_preserved(before, after)


@pytest.mark.parametrize("field", ["schema_hash", "ttmlir_git_hash"])
def test_wrong_original_metadata_fails(field):
    document = original()
    document[field] = "wrong"
    with pytest.raises(ValueError):
        migration.validate_original(document)


def test_generated_schema_must_match_compiler_before_decode(tmp_path):
    destination = tmp_path / "generated"
    def fake_flatc(*args, **kwargs):
        (destination / "system_desc_bfbs_generated.h").write_text("wrong schema")
    with patch.object(migration, "run", side_effect=fake_flatc):
        with pytest.raises(ValueError, match="differs from installed compiler"):
            migration.generate_schema_hash(tmp_path / "flatc", tmp_path, destination)


def test_checked_in_schema_bundle_is_exact(tmp_path):
    bundle = PATH.parents[1] / "configs/descriptor-schemas"
    roots = migration.prepare_schemas(bundle, tmp_path)
    assert set(roots) == {"old", "new"}


def test_tampered_schema_fails_before_conversion(tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "old").mkdir(parents=True)
    (bundle / "old/system_desc.fbs").write_text("tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        migration.prepare_schemas(bundle, tmp_path / "work")
