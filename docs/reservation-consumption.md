# Reservation consumption wire encoding

`laya_tt.ledger_protocol.encode_consumption` maps already verified admission claims,
trusted prepared row/token counts, and independently observed loaded-runtime metadata
to the closed reservation-consumption v1 payload. It does not authenticate a grant,
contact a platform endpoint, or authorize execution.

The payload preserves the original reservation ceilings and deadline. Actual
prepared counts are separate fields and must fit those ceilings. The binding retains
tenant, key, request, attempt, reservation, usage-event, request digest and policy
identities. Runtime metadata must agree with the claims' model, checkpoint and
generation; its revision and backend come from trusted bootstrap/readback.

All wire containers are JSON objects. Unknown fields, host identity overrides,
credentials and customer content are excluded. Integer counts must be actual
integers, not booleans or integral floats. Deadlines fit the durable ledger's signed
64-bit range. Structural schema validation does not replace the producer/receiver
checks that prepared counts fit the original reservation.

The platform-owned receiver must derive host identity from authentication, compare
the entire independently persisted reservation, check database time and consume once
before returning committed success. Lost acknowledgments require reconciliation;
repeating consumption is not an execution retry policy. The runtime's local transport
and platform grant verifier remain explicit composition responsibilities.

`schemas/fixtures/reservation-consumption.json` is produced by the real encoder
under synthetic context/runtime inputs. It preserves a ceiling of 20 encoded tokens
and records 17 prepared tokens. It is portable wire evidence, not model execution,
physical acceptance, or a live reservation. The test freezes its exact bytes.
