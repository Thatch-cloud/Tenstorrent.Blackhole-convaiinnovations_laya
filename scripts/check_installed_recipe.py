"""Check the installed package contains only standalone model helpers."""
from importlib.metadata import files
from laya_tt import cpu_backend, integrity, reference

expected = {"laya_tt/" + name + ".py" for name in
            ("__init__", "cpu_backend", "integrity", "reference")}
installed = {str(path).replace("\\", "/") for path in files("laya-tt")
             if str(path).replace("\\", "/").startswith("laya_tt/")
             and "__pycache__" not in str(path)}
assert installed == expected, "Unexpected files in standalone model package"
assert callable(cpu_backend.load_cpu_backend)
assert callable(integrity.verify_loaded_state)
assert len(reference.UPSTREAM_COMMIT) == 40
