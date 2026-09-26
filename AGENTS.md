# Public repository boundary

This is a public, standalone Laya model port. Keep changes limited to pinned
public model inputs, CPU reference validation, compiler experiments, model
execution, synthetic test cases and reproducible setup instructions.

Do not add organization-specific hosting, authentication, tenancy, billing,
control-plane protocols, deployment manifests, runner configuration, operational
inventories, private repository references, credentials or real customer data.
Keep deployment integration work in an appropriately private repository.

Generated host logs, environment captures and compiler output belong under
ignored `artifacts/`. Review any proposed fixture for identifying metadata and
archive contents before committing it. Use generic hardware identifiers in tests.
Never publish a raw operational capture as model evidence.

Preserve checkpoint and fixture hashes. Validate model changes with the relevant
tests and validate packaging from a clean build directory. A successful offline
compile is not physical inference or numerical acceptance.
