# Decision grant verification profile v1

`DecisionGrantVerifier` is an explicit callable for `RuntimeApplication.verify_grant`.
Bootstrap supplies an owned snapshot of 1-16 pinned raw Ed25519 public keys, indexed
by safe ASCII key IDs, plus the expected issuer, stable host identity, complete
loaded runtime identity and policy revision. There is no network key lookup,
node credential, private key, signer or permissive default in the runtime.
The platform must provision and rotate this trust through its existing key ownership
infrastructure. Replacing the verifier replaces its trusted key set; the library
does not implement live revocation distribution.

Opaque grants use compact JWS: three unpadded canonical base64url segments joined
by periods. Total size is at most 12 KiB. The closed protected header contains
exactly `alg: "Ed25519"`, `typ: "thatch-decision-grant+jws"`, and `kid`.
The fully specified algorithm identifier follows
[RFC 9864 section 2.2](https://www.rfc-editor.org/rfc/rfc9864.html#section-2.2).
The signature covers the original ASCII `protected.payload` bytes; JSON is never
re-serialized for verification. EdDSA, unsigned tokens, other algorithms, key URLs,
embedded keys, detached payloads and header extensions are rejected.

The UTF-8 payload is a closed object with exactly these fields:

- `schema_version`: `"1"`.
- `issuer` and `host_id`: exact trusted assignment strings.
- `runtime`: the full execution-receipt runtime object, matching bootstrap.
- `context`: the existing closed admission-context object.

Duplicate fields, nonfinite values, malformed encodings, wrong signatures and
unknown keys fail closed. Context clocks/counts must be strict integers; the grant
must currently satisfy issued-at <= now < deadline, without clock-skew grace.
Model/checkpoint/generation and policy must match the trusted assignment. All errors
expose fixed text without grant content. Cryptographic verification uses
[cryptography's Ed25519 verifier](https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ed25519/).

Admission still binds `request_sha256` to the actual request, prepares and checks
work limits, consumes the durable reservation exactly once, and records the local
intent before execution. A valid signature alone never permits execution. The
issuer must commit the matching reservation before returning a grant, use the same
stable host and runtime binding, and preserve uncertain reservations for recovery.

This implements the runtime verifier and a proposed versioned interoperability
profile. Platform signing, Rust cross-language fixture verification, key provisioning,
and end-to-end authenticated gateway composition remain required before activation.
The tests' deterministic signing seed is public test material only.
