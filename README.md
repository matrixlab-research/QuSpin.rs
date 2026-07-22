# QuSpin.rs

`quspin` is a Rust-native exact-diagonalization toolkit being built from real
many-body workflows rather than by transliterating Python or Julia syntax.

The frozen clean-room task documents are published separately:

- [motivation](https://github.com/matrixlab-research/quspin-rust-task/blob/main/MOTIVATION.md)
- [API contract](https://github.com/matrixlab-research/quspin-rust-task/blob/main/CONTRACT.md)
- [visible examples](https://github.com/matrixlab-research/quspin-rust-task/blob/main/TESTS.md)

The implementation follows one narrow waist: bases define physical state and
local-operator semantics, while stored and matrix-free linear maps expose the
same rectangular `shape + apply` interface. This supports square Hamiltonians,
cross-sector probes, and open-system generators without model-specific solver
paths.

Status: initial implementation in progress. API presence alone is not a claim
of Python/Julia parity; numerical completion is tracked by the private paper
workflow verifier.
