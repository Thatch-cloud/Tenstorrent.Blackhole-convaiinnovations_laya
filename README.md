# Laya on Tenstorrent Blackhole

Standalone, experimental recipe for running [Laya](https://github.com/NandhaKishorM/laya)
on one Tenstorrent P150 card. This repository contains model preparation, CPU
reference validation, and TT-XLA compilation experiments.

**Status:** CPU references and offline compilation have passed. Physical TT
inference, numerical parity, memory usage, and latency have not been accepted.
This is not yet a working accelerated inference release.

## Prepare the reference

Python 3.11 or newer is required. The Windows CPU environment is pinned separately
from the Linux compiler environment.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-reference-windows-py311.lock
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
git clone --no-checkout https://github.com/NandhaKishorM/laya.git .cache/upstream/laya
git -C .cache/upstream/laya checkout --detach 970dc8c5f63d7b886a68409493f37d569424f933
.\.venv\Scripts\python.exe scripts/download_checkpoint.py
.\.venv\Scripts\python.exe scripts/build_reference.py --output artifacts/reference/cpu
.\.venv\Scripts\python.exe -m pytest -q
```

The checkpoint lock pins revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`
and every file digest. Weights are downloaded from the upstream publisher and are
not included. Source, checkpoint, tokenizer, and loaded weights must pass validation;
do not silently repin changed files or overwrite earlier experiment output.

## Compile for P150

Use Linux with the matched packages in `requirements-compiler-linux-py312.lock`.
The manual `Independent offline option-bucket compilation` workflow provides a
reproducible device-free setup. It can compile the original option batch or five
single-row profiles (32, 64, 128, 256, and 512 tokens). FP32 source is the default;
the mixed BF16/FP32 policy is experimental and has no hardware parity acceptance.

See [compiler instructions](docs/compiler-spike.md), [shape and precision probes](docs/model-experiments.md),
[descriptor migration](docs/descriptor-migration.md), [CPU reference](docs/reference.md),
and [loaded-state validation](docs/integrity.md).

Offline compilation uses a saved generic descriptor and produces dummy outputs.
It establishes compiler feasibility only. Before physical execution, verify exclusive
access to the selected board and the installed runtime's device-discovery behavior;
see the [device visibility audit](docs/device-visibility-audit.md). No reset or
firmware change is part of the recipe.

## Repository scope

Only standalone model code, public upstream dependencies, synthetic fixtures,
and reproducible model experiments belong here. Deployment integrations and
operational inventories are outside this repository's scope. Keep generated
logs, host captures, model weights, and compiler artifacts under ignored `artifacts/`.
Review generated files before sharing them; they can contain machine paths and
other environment details.
