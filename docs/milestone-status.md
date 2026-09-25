# First milestone evidence and remaining work

This status separates local evidence from compiler, accelerator and deployment
acceptance. It is not a claim that the complete single-P150A service is ready.

| Requirement | Current evidence | Remaining gate |
|---|---|---|
| Pinned English source and model | Immutable source/checkpoint lock plus SHA256 for all five checkpoint assets | Reproduce under accelerator toolchain |
| Reproducible CPU reference | Original two CPU captures match exactly; fresh GitHub and WSL full-model checks also pass loaded-state integrity, exact inputs/answers, and fixed CPU logit tolerances | Intermittent local loaded-state failure remains unexplained; derived rotary buffers and representative application coverage remain |
| Typed semantics and action outputs | Real fixtures cover choice, score, noul, single-option, mixed widths, batching, truncation, empty questions and temperature buckets | Physical device probability/decision parity, approved tolerances |
| Strict full-model export | Five actual CPU graphs export and match the checked baseline with zero observed output error | All five FP32-source calls compiled offline (lowered IR includes BF16); prior local integrity failure remains unexplained; physical TT execution remains unverified |
| Shared runtime admission | Exact-byte payload binding, injected grant verifier and durable reservation hook, bounded payload/token/work admission | Real platform credential issuer and ledger implementation |
| Serialized shared worker | Tenant rotation, queue budgets, deadlines, cancellation and drain ownership tests | Local journal exposes interrupted work and prevents replay; device ownership recovery remains |
| Runtime composition | Actual CPU backend through admission/worker/response path with two tenants using identical IDs | Production startup and deployment wiring |
| Internal HTTP transport | In-process ASGI tests cover bounded fragmented bodies, opaque grant forwarding, readiness, event-loop responsiveness and disconnect ownership | Real host dispatch and listener deployment |
| Runtime readiness | Required model self-test, failed/draining states and preparation concurrency bound | Actual accelerator startup/warmup and hardware observations |
| Installed contracts | Wheel contains all three schemas and isolated wheel import validates them | Versioned release and consumer dependency pins |
| Usage | Measured local queue/execution times, question rows and encoded tokens; no invented accelerator time | Optional durable local journal/outbox now tested; authenticated delivery, authoritative accounting and billing policy remain |
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
