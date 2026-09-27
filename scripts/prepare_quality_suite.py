"""Create deterministic, hash-verified public benchmark inputs; no model loading."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "ag_news": ["world", "sports", "business", "science and technology"],
    "emotion": ["sadness", "joy", "love", "anger", "fear", "surprise"],
    "sst5": ["very negative", "negative", "neutral", "positive", "very positive"],
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def convert(identifier, index, row):
    if identifier == "boolq":
        assert isinstance(row["answer"], bool)
        state = {"passage": row["passage"], "question": row["question"]}
        question = {"type": "noul", "instructions": "Using the passage, answer the question with yes or no."}
        expected = row["answer"]
    else:
        labels = LABELS[identifier]
        label = row["label"]
        assert isinstance(label, int) and 0 <= label < len(labels)
        state = {"text": row["text"]}
        if identifier == "sst5":
            question = {"type": "score", "instructions": "Rate the sentiment of the text.", "criteria": labels}
            expected = label
        else:
            instruction = "Classify the news topic." if identifier == "ag_news" else "Identify the emotion expressed in the text."
            question = {"type": "choice", "instructions": instruction, "criteria": dict.fromkeys(labels)}
            expected = labels[label]
    assert all(isinstance(value, str) and value for value in state.values())
    return {"id": f"{identifier}:{index}", "dataset": identifier, "source_row": index,
            "state": state, "questions": {"decision": question}, "expected": {"decision": expected}}


def prepare(lock_path, cache, output, per_dataset=0, download=False):
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    cache.mkdir(parents=True, exist_ok=True)
    records, sources = [], []
    for spec in lock["datasets"]:
        path = cache / (spec["id"] + Path(spec["file"]).suffix)
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing {path}; use --download to fetch pinned public data")
            url = f'https://huggingface.co/datasets/{spec["repository"]}/resolve/{spec["revision"]}/{spec["file"]}'
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if digest(data) != spec["sha256"]:
                raise ValueError(f'Hash mismatch: {spec["id"]}')
            path.write_bytes(data)
        data = path.read_bytes()
        if digest(data) != spec["sha256"]:
            raise ValueError(f'Hash mismatch: {spec["id"]}')
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            rows = pq.read_table(path).to_pylist()
        else:
            rows = [json.loads(line) for line in data.decode().splitlines() if line.strip()]
        if len(rows) != spec["rows"]:
            raise ValueError(f'Row count mismatch: {spec["id"]}')
        selected = [convert(spec["id"], i, row) for i, row in enumerate(rows)]
        # Fixed hash ordering avoids favouring source order or model outcomes.
        selected.sort(key=lambda row: digest(("laya-quality-v1:" + row["id"]).encode()))
        if per_dataset:
            selected = selected[:per_dataset]
        records.extend(selected)
        sources.append({"id": spec["id"], "source_rows": len(rows), "selected_rows": len(selected), "sha256": spec["sha256"]})
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    canonical_lock = json.dumps(lock, sort_keys=True, separators=(",", ":")).encode()
    manifest = {"schema_version": 1, "canonical_lock_sha256": digest(canonical_lock), "cases_sha256": digest(payload),
                "rows": len(records), "per_dataset": per_dataset, "selection": "sha256(laya-quality-v1:<dataset>:<source_row>)",
                "sources": sources, "model_executed": False, "physical_acceptance": False}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=ROOT / "configs/quality-datasets.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "artifacts/quality-sources")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/quality/cases.jsonl")
    parser.add_argument("--per-dataset", type=int, default=0, help="0 uses every row; nonzero is diagnostic only")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.per_dataset < 0:
        parser.error("--per-dataset must be nonnegative")
    print(json.dumps(prepare(args.lock, args.cache, args.output, args.per_dataset, args.download), indent=2))
