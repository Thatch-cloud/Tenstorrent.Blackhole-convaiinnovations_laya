"""Load versioned contracts from the installed distribution."""
import json
from importlib.resources import files

_NAMES = frozenset(("decision-request", "decision-response", "admission-context", "execution-receipt"))

def load_schema(name: str) -> dict:
    if name not in _NAMES:
        raise ValueError("unknown contract schema")
    return json.loads(files("laya_tt").joinpath("schemas", name + ".schema.json").read_text(encoding="utf-8"))
