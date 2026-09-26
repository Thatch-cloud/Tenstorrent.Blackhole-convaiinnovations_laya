"""Run with python -I in a fresh wheel-only environment; no source path injection."""
from jsonschema import Draft202012Validator
from laya_tt.asgi import DecisionASGI
from laya_tt.bootstrap import RuntimeApplication
from laya_tt.grants import DecisionGrantVerifier
from laya_tt.contracts import load_schema

for name in ("decision-request", "decision-response", "admission-context",
             "reservation-consumption", "execution-receipt"):
    Draft202012Validator.check_schema(load_schema(name))

assert callable(DecisionASGI)
assert callable(RuntimeApplication)
assert callable(DecisionGrantVerifier)
print("Installed runtime composition and all five packaged schemas validated.")
