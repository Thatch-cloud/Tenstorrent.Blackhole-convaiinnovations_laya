"""Hash an already downloaded immutable checkpoint; never downloads weights."""
import argparse
import json
from pathlib import Path
import re
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from laya_tt.reference import UPSTREAM_COMMIT, checkpoint_files, resolve_path, validate_manifest
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--snapshot-dir", default=".cache/checkpoints/laya-english")
    parser.add_argument("--upstream-path", default=".cache/upstream/laya")
    parser.add_argument("--upstream-commit", default=UPSTREAM_COMMIT)
    parser.add_argument("--output", default="configs/checkpoint-lock.json")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision requires a full immutable commit SHA")
    manifest = {"schema_version": 1,
                "upstream": {"commit": args.upstream_commit, "path": args.upstream_path},
                "checkpoint": {"repo_id": args.repo_id, "revision": args.revision,
                               "path": args.snapshot_dir,
                               "files": checkpoint_files(resolve_path(ROOT, args.snapshot_dir))},
                "runtime": {"device": "cpu", "dtype": "float32"}}
    validate_manifest(manifest, ROOT)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Pinned {len(manifest['checkpoint']['files'])} artifacts in {output}")
if __name__ == "__main__":
    main()
