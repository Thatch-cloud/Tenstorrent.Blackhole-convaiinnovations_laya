"""Run with python -I in a fresh wheel-only environment; no source path injection."""
from jsonschema import Draft202012Validator
from laya_tt.asgi import DecisionASGI
from laya_tt.contracts import load_schema

for name in ("decision-request", "decision-response", "admission-context"):
    Draft202012Validator.check_schema(load_schema(name))

assert callable(DecisionASGI)
print("Installed runtime imports and all three packaged schemas validated.")
