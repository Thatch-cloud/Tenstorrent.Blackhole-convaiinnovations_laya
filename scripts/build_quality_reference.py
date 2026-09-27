"""Capture pinned CPU/FP32 tensors and native answers for a prepared quality suite."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from laya_tt.reference import capture_reference, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "artifacts/quality/pilot.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/quality-reference")
    args = parser.parse_args()
    manifest = json.loads(args.cases.with_suffix(".manifest.json").read_text())
    if sha256_file(args.cases) != manifest["cases_sha256"]:
        raise ValueError("Quality input hash mismatch")
    records = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines()]
    if len(records) != manifest["rows"] or not records:
        raise ValueError("Quality input count mismatch")
    cases = []
    identities = set()
    for row in records:
        identifier = row["id"]
        if identifier != f'{row["dataset"]}:{row["source_row"]}' or identifier in identities:
            raise ValueError("Invalid or duplicate quality identity")
        identities.add(identifier)
        # Filesystem-safe identifiers also make this recipe usable on Windows.
        safe_id = "quality-" + identifier.replace(":", "-")
        if any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in safe_id):
            raise ValueError("Unsafe quality identity")
        cases.append({"id": safe_id, "quality_id": identifier, "state": row["state"],
                      "questions": row["questions"], "expected": row["expected"]})
    args.output.mkdir(parents=True, exist_ok=False)
    converted = args.output / "capture-inputs.json"
    converted.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = capture_reference(ROOT / "configs/checkpoint-lock.json", converted,
                               args.output / "capture", ROOT)
    summary = {"prepared_cases_sha256": manifest["cases_sha256"], "rows": len(cases),
               "selection": manifest, "reference_sha256": sha256_file(args.output / "capture/reference.json"),
               "captured_cases": len(report["cases"]), "physical_acceptance": False,
               "quality_qualification": False}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
