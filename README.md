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

Status: the public release-mode gate executes all 12 frozen paper workflows,
including symmetry reduction, sparse shift-invert, Floquet evolution,
open-system dynamics, and cross-sector spectroscopy. API presence and this
public regression suite are still not, by themselves, a claim of Python/Julia
parity; independent numerical observations and benchmark records remain the
responsibility of the private paper-workflow verifier.
