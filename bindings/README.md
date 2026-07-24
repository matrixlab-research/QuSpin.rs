# QMBED language bindings

Both language packages are thin request builders over `qmbed-capi`, which
links the same Rust core as native applications. The C boundary accepts one
typed JSON schema for built-in bases, `OpProduct` terms, materialization format,
and `eigsh` options. It returns dimensions, convergence evidence, eigenvalues,
residuals, and optionally eigenvectors.

- `python/` exposes the native `qmbed` module and `qmbed.compat.quspin`.
- `julia/` exposes `QMBED` and `QMBED.Compat.QuSpin`.
- `capi/` owns serialization and the only unsafe pointer boundary.

Site indices are zero based in all three languages. Compatibility operator
strings are parsed in the language adapter and sent as typed local actions;
they do not select a separate Rust assembler.
