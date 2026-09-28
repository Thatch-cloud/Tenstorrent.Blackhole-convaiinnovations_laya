"""Compatibility import for model profile probes."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from laya_tt.shape_profiles import BUCKETS, MARKERS, profile_rows
