"""Build authentic upstream CPU reference outputs."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from laya_tt.reference import capture_reference
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/checkpoint-lock.json")
    parser.add_argument("--cases", default="configs/reference-cases.json")
    parser.add_argument("--output", default="artifacts/reference/cpu")
    args = parser.parse_args()
    report = capture_reference(ROOT / args.manifest, ROOT / args.cases, ROOT / args.output, ROOT)
    print(f"Captured {len(report['cases'])} cases in {args.output}")
if __name__ == "__main__":
    main()
