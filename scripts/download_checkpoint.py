"""Download only the pinned English checkpoint; no latest/default resolution."""

import argparse
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851")
    parser.add_argument("--output", type=Path, default=Path(".cache/checkpoints/laya-english"))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("revision must be an immutable 40-character commit SHA")
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="convaiinnovations/laya",
        revision=args.revision,
        local_dir=args.output,
        allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/*"],
    )
    print(f"Downloaded pinned English checkpoint {args.revision} to {args.output}")


if __name__ == "__main__":
    main()
