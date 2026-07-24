# Release process

QMBED has one Rust core and three versioned distribution surfaces. A release
must keep these versions equal:

- `Cargo.toml`
- `bindings/capi/Cargo.toml`
- `bindings/python/pyproject.toml`
- `bindings/julia/Project.toml`

Before tagging:

1. update all four versions;
2. merge only after CI and language-bindings checks pass on `main`;
3. run the QMBED Benchmark Rust contract and a paper-tier same-runner benchmark;
4. tag the verified `main` commit as `vX.Y.Z`.

The tag workflow rechecks version equality, Rust tests and package assembly,
the C boundary, and both language bindings before it creates the GitHub
release. Publishing to crates.io, PyPI, or Julia General remains a separate
explicit action so a Git tag cannot publish packages with external registry
credentials.
