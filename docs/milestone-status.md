# First milestone evidence and remaining work

This status separates local evidence from compiler, accelerator and deployment
acceptance. It is not a claim that the complete single-P150A service is ready.

| Requirement | Current evidence | Remaining gate |
|---|---|---|
| Pinned English source and model | Immutable source/checkpoint lock plus SHA256 for all five checkpoint assets | Reproduce under accelerator toolchain |
| Reproducible CPU reference | Original two CPU captures match exactly; fresh GitHub and WSL full-model checks also pass loaded-state integrity, exact inputs/answers, and fixed CPU logit tolerances | Intermittent local loaded-state failure remains unexplained; derived rotary buffers and representative application coverage remain |
| Typed semantics and action outputs | Real fixtures cover choice, score, noul, single-option, mixed widths, batching, truncation, empty questions and temperature buckets | Physical device probability/decision parity, approved tolerances |
| Strict full-model export | Five actual CPU graphs export and match the checked baseline with zero observed output error | All five FP32-source calls compiled offline (lowered IR includes BF16); prior local integrity failure remains unexplained; physical TT execution remains unverified |
| Finite single-row shapes | [CPU experiment](shape-profile-experiment.md) preserves decoded answers across six fixtures; full-model padding sweep passes at widths 32, 64, 128, 256 and 512 with 64 marker slots | Content-length and marker-limit quality coverage, exact-profile compilation, precision acceptance and physical parity |
| Shared runtime admission | Exact-byte payload binding, pinned-key Ed25519 issuer/host/runtime/policy verification, journaled reservation consumption, bounded payload/token/work admission | Production authority, credential ownership and service composition |
| Serialized shared worker | Tenant rotation, queue budgets, deadlines, cancellation and drain ownership tests | Local journal exposes interrupted work and prevents replay; device ownership recovery remains |
| Runtime composition | CPU backend contract tests and real pinned model through signed admission, ASGI, worker, Unix ledger peer and durable outbox; separate synthetic multi-tenant tests | Combined platform isolation/accounting acceptance, production startup and deployment wiring |
| Internal HTTP transport | In-process ASGI tests cover bounded fragmented bodies, opaque grant forwarding, readiness, event-loop responsiveness and disconnect ownership | Real host dispatch and listener deployment |
| Runtime readiness | Required model self-test, failed/draining states and preparation concurrency bound | Actual accelerator startup/warmup and hardware observations |
| Installed contracts | Wheel contains all five schemas and isolated wheel import validates runtime composition | Versioned release and consumer dependency pins |
| Usage | Measured local queue/execution times, question rows and encoded tokens; durable journal/outbox with exact acknowledgment; no invented accelerator time | Production authenticated delivery, authoritative accounting and billing policy |
| Single-board ownership | Read-only hardware inventory completed | Enforced allocation and version-specific initialization isolation |
| Multi-tenant platform release | Initial protocol compatibility work is separate | Gateway, Management dispatch, Compute lifecycle, isolation/recovery tests and deployed acceptance |

The compatibility `/v1/systemone` alias is not implemented. Native requests do
not silently truncate or select an unknown checkpoint. The CPU reference is an
explicit development backend and is never an automatic accelerator fallback.

Initial Windows captures without loaded-state verification are quarantined and
excluded from accepted fixtures. Exact repeatability after the integrity gate
does not diagnose the original mismatch's cause. The upstream 11-or-more-option
temperature is clamped; affected confidence is uncalibrated.

No TT latency, throughput, capacity, tenant hardware isolation or deployed
service claim follows from these local checks. The BF16 one-card graph,
end-to-end benchmarks, immutable serving image and canary/rollback evidence
remain outstanding.


## Published CPU contract check

GitHub Actions run [36093507962](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-convaiinnovations_laya/actions/runs/36093507962)
passed on commit `604657517f411187d664446569f5e63409dd54b5` for Python 3.11
and 3.12 on Ubuntu 24.04. Each job ran 161 tests with 2 optional model-backed
checks skipped, plus 92 passing subtests, then validated an isolated wheel install.
This checks the committed fixture bytes, runtime contracts and package resources;
it does not download or reload the full checkpoint and is not hardware acceptance.


## Independent full-model CPU evidence

[Hosted run 36094671855](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-convaiinnovations_laya/actions/runs/36094671855)
performed a fresh pinned download and full-Agent capture on commit `c962f22`,
then passed the fixed CPU comparison policy. A separate local WSL full-Agent
capture also passed. Retained reports and captures are indexed in
[evidence/reference/independent-host-summary.json](../evidence/reference/independent-host-summary.json).
The original Windows baseline uses Torch 2.14.0; these fresh Linux captures use
Torch 2.11.0+cpu. Decoded answers and input tensors match exactly; floating outputs
are compared numerically and are not claimed bit-identical across these toolchains.

Three bounded local tensor-conversion/copy sweeps also matched. Neither those
sweeps nor the successful full-model captures establish the cause of the earlier
local loading failure. The compiler matrix failure remains preserved separately.

## Complete offline reference-shape coverage

[Independent compiler run 36095766121](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-convaiinnovations_laya/actions/runs/36095766121) passed on `ff79175`, compiling option-temperature-buckets `[3,84]` after exact loaded-state verification. Parent review verified the report, log, three compiled artifact hashes and committed harness hashes. [Combined evidence](../evidence/compiler/offline-fp32-complete-coverage.json) now covers all five reference forward graphs; the earlier failure evidence is unchanged. No TT device was exposed, no device output was compared, and physical acceptance remains false.

CPU contract CI [36095760279](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-convaiinnovations_laya/actions/runs/36095760279) passed on the same commit for Python 3.11 and 3.12: each ran 199 tests, 2 optional skips and 92 passing subtests, followed by isolated wheel validation.

## Mixed-precision CPU observation and runtime follow-up

The [explicit BF16/FP32 CPU candidate](bf16-cpu-experiment.md) passed exact state and activation-dtype checks, retaining discrete choices and order on the six fixtures. Calibrated option probabilities changed by up to 0.0104 and scores by 0.0164. Saturated action probabilities cannot validate action quality. This candidate is not promoted and has not been compiled or executed on TT.

Runtime readback and unreconciled-startup protection were published in `e1a197c`; [CI run 36096388343](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-convaiinnovations_laya/actions/runs/36096388343) passed. Runtime facts still require collection and production composition in Compute/Management. A local service test now verifies that multiple authenticated key fixtures share a tenant queue budget and fair rotation while durable receipts retain individual key attribution; this is not a deployed gateway test.

## Current runtime composition evidence

At `e136b60b3e4fe776a533cd1deb5b547b8e3a1b09`, [contract CI
36235661495](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-convaiinnovations_laya/actions/runs/36235661495)
passed on Python 3.11 and 3.12: each job ran 355 tests, 3 optional skips and
107 passing subtests, then validated the installed runtime and all five schemas.

[Independent CPU run 36235662131](https://github.com/Thatch-cloud/Tenstorrent.Blackhole-convaiinnovations_laya/actions/runs/36235662131)
downloaded the pinned assets, passed the full reference comparison, then ran the
real CPU runtime composition test. Its startup self-test and admitted ASGI request
matched the committed mixed-question-widths answers, including choice, score,
noul, action outputs and the one-option case. It verified the signed grant, actual
prepared counts, Unix peer credentials, original consumption/receipt bytes and
outbox state after reopening the journal before and after acknowledgment.

The ledger peer in that public test is a recording fixture, not a platform quota
authority. This evidence does not establish the combined deployed platform path,
cross-replica accounting, device ownership or TT numerical/performance acceptance.
See [runtime composition](runtime-composition.md) for the reproducible command.
