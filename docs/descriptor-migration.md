# Offline descriptor migration

The official P150 fixture in TT-XLA commit `873c53c4ff84bd91112a1f43e788cfed75aaa914` was produced by older TT-MLIR `464a5f341908d5a3f790e697be5ffa6fd3fb6f32`. Its schema hash does not match the installed nightly compiler. The compiler correctly rejects it before model lowering.

`scripts/migrate_system_descriptor.py` is a local migration utility, not an upstream-supported converter. It never imports the TT runtime or queries hardware. The original fixture is immutable; the result is a **migrated generic descriptor for offline compilation**, not current card inventory or physical acceptance evidence.

Inputs are pinned byte-for-byte: the original fixture, three original schema files, three target schema files, and FlatBuffers source commit `fb9afbafc7dfe226b9db54d4923bfb8839635274`. This FlatBuffers commit comes from the target MLIR `env/CMakeLists.txt`. Build `flatc` from that clean checkout. Its binary hash and version are recorded, and the tool independently reproduces the target compiler's generated schema-header hash using the exact options from `cmake/modules/BuildFlatbuffers.cmake`. A mismatch stops conversion.

The complete schema delta consists of two enums added to `types.fbs`: `MoEActivationFunction` and `DataMovementProcessor`. Neither changes the system or chip descriptor tables. `system_desc.fbs` and `version.fbs` are byte-identical between source revisions.

The utility decodes the original using its own schema with explicit defaults, re-encodes with the target schema, and decodes the result under both schemas. Both complete decoded roots must equal the original except for `schema_hash`. Original producer/version metadata remains unchanged, and the sidecar records the migration separately. No binary hash bytes are patched and the compiler's validator remains enabled.

Example after building the pinned FlatBuffers checkout:

```bash
python scripts/migrate_system_descriptor.py \
  --input configs/compiler/p150_system_desc.ttsys \
  --flatc /opt/laya-flatbuffers-fb9afb/build/flatc \
  --flatbuffers-source /opt/laya-flatbuffers-fb9afb \
  --output artifacts/descriptor-migration/p150-nightly
```

Adjust the original fixture path to its checked-in location. The output directory must not exist. Output includes the derived descriptor, original and migrated decoded JSON, and provenance. The new descriptor requires separate review and an explicit offline-only harness pin before use; it must never be accepted for physical inference or identified as a fresh runtime capture.

Source schemas are from [original MLIR](https://github.com/tenstorrent/tt-mlir/tree/464a5f341908d5a3f790e697be5ffa6fd3fb6f32/include/ttmlir/Target/Common) and [target MLIR](https://github.com/tenstorrent/tt-mlir/tree/e2b21a721955b2b68540d9e6879a215ce3dcfae5/include/ttmlir/Target/Common). The [schema guard](https://github.com/tenstorrent/tt-mlir/blob/e2b21a721955b2b68540d9e6879a215ce3dcfae5/lib/Dialect/TTCore/IR/TTCoreOpsTypes.cpp#L367) compares against the [generated header digest](https://github.com/tenstorrent/tt-mlir/blob/e2b21a721955b2b68540d9e6879a215ce3dcfae5/tools/scripts/sha256-include-gen.py).

The checked migration succeeded with flatc24.3.25 built at the pinned revision.
The derived artifact and original/migrated JSON plus provenance are retained in
`configs/compiler/migrated-p150/`. It reproduces the expected compiler schema
hash and passes complete-root equality with both old and new schema decoders.
Its descriptor SHA256 is
`6028485af0f8c233185b87277f36061c2460db606f14a47bcb8284f5af5f56f3`.
This result validates the offline format migration, not hardware state or inference.
