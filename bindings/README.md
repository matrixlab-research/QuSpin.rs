# QMBED language bindings

Both language packages are thin request builders over `qmbed-capi`, which
links the same Rust core as native applications. The C boundary accepts one
typed JSON schema for built-in bases, `OpProduct` terms, materialization format,
and `eigsh` options. It returns dimensions, convergence evidence, eigenvalues,
residuals, and optionally eigenvectors.

Long-lived frontend objects use the same schema through a persistent-model
protocol: `create_model` returns an opaque decimal handle, model operations
reuse that handle, and `release_model` ends its lifetime. Handles are unique,
safe to use concurrently, and never expose a Rust address. Release is
deterministic when a frontend provides `close()` and is also backed by a
best-effort finalizer. A command which uses a released or unknown handle fails
explicitly. The Rust `PackedEdModel` also caches one assembled operator per
storage format, so repeated solver and export calls reuse both the basis and
the assembled matrix.

- `python/` exposes the native `qmbed` module and the versioned
  `qmbed.compat.quspin` migration surface.
- `julia/` exposes only the native `QMBED` API.
- `capi/` owns serialization and the only unsafe pointer boundary.

Site indices are zero based in all three languages. Python compatibility
operator strings are parsed in the adapter and sent as typed local actions;
they do not select a separate Rust assembler. Julia callers construct
`OpProduct` and `OperatorSpec` values directly.

General packed bases use a serializable lattice-symmetry schema rather than
frontend callbacks. Each generator specifies a bijection of source sites,
optional per-site permutations of local states, and a character sector. Rust
validates the map, derives its finite period, and computes fermionic exchange
phases. The same representation therefore covers translations, reflections,
local spin inversion, and higher-dimensional lattice permutations.

## Python compatibility contract

The directory `python/compat_tests/quspin-1.0.1/` is a byte-for-byte snapshot
of the 73 official Python tests from QuSpin 1.0.1 at commit
`5bf9e5b266e6d8b70e5cf5973c7c7d59d62e412f`. Its upstream BSD-3-Clause license,
file hashes, and an exhaustive compatibility status are committed beside the
tests.

CI runs every test marked `passing` without modifying its source. Tests whose
required object protocol has not yet been implemented remain explicitly
listed as `unsupported`; they are not silently skipped. The snapshot and
classification can be checked locally with:

```bash
python ci/freeze_upstream_quspin_tests.py --check
```
