# First milestone evidence and remaining work

This status separates local evidence from compiler, accelerator and deployment
acceptance. It is not a claim that the complete single-P150A service is ready.

| Requirement | Current evidence | Remaining gate |
|---|---|---|
| Pinned English source and model | Immutable source/checkpoint lock plus SHA256 for all five checkpoint assets | Reproduce under accelerator toolchain |
| Reproducible CPU reference | Two independent processes pass exact loaded-state checks; 35 tensors and every answer match exactly | Derived rotary buffers and representative application dataset coverage |
| Typed semantics and action outputs | Real fixtures cover choice, score, noul, single-option, mixed widths, batching, truncation, empty questions and temperature buckets | Physical device probability/decision parity, approved tolerances |
| Strict full-model export | Five actual CPU graphs export and match the checked baseline with zero observed output error | Single-option FP32 graph compiled offline after verified descriptor migration; remaining shapes, BF16 and physical TT execution are unverified |
| Shared runtime admission | Exact-byte payload binding, injected grant verifier and durable reservation hook, bounded payload/token/work admission | Real platform credential issuer and ledger implementation |
| Serialized shared worker | Tenant rotation, queue budgets, deadlines, cancellation and drain ownership tests | Host crash recovery and device claims |
| Runtime composition | Actual CPU backend through admission/worker/response path with two tenants using identical IDs | Production startup and deployment wiring |
| Internal HTTP transport | In-process ASGI tests cover bounded fragmented bodies, opaque grant forwarding, readiness, event-loop responsiveness and disconnect ownership | Real host dispatch and listener deployment |
| Runtime readiness | Required model self-test, failed/draining states and preparation concurrency bound | Actual accelerator startup/warmup and hardware observations |
| Installed contracts | Wheel contains all three schemas and isolated wheel import validates them | Versioned release and consumer dependency pins |
| Usage | Measured local queue/execution times, question rows and encoded tokens; no invented accelerator time | Durable receipts/replay/accounting and explicit billing policy |
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
