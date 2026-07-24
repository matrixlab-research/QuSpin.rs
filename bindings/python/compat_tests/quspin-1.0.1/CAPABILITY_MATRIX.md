# QuSpin 1.0.1 compatibility capability matrix

This matrix is derived from the unchanged 73-file upstream test snapshot. A
row is complete only when the original test executes through the public Python
compatibility package and the exercised behavior is covered by Rust tests.

## Implemented foundation

| Capability | Rust core | Language-neutral boundary | Upstream evidence |
|---|---|---|---|
| Runtime-selected packed basis | `PackedBasis` preserves concrete spin, boson, spinless-fermion, and spinful-fermion semantics | `describe_basis` | `test_version.py`, `test_boson_vs_ho.py`, `test_higher_spin.py` |
| Reusable ED model | `PackedEdModel` owns basis, typed terms, checks, materialization, `eigh`, and `eigsh` | `materialize`, `eigh`, `eigsh` commands | boson/HO full-spectrum equality |
| Explicit basis-vector ordering | `PackedBasis::reversed` permutes `state`, `index`, and transition rows together | `reverse` basis option | all higher-spin `x/y/z/I` products |
| Explicit site convention | `OperatorSpec::with_site_permutation` relabels every coupling through one validated bijection | `site_permutation` command field | all higher-spin multi-site tensor products |
| Python drop-in namespace | Rust-backed `quspin.basis` and `quspin.operators` modules | same C ABI used by native Python and Julia | 3/73 files passing unchanged |

## Remaining foundation gaps

| Priority | Required behavior | Rust/core gap | Boundary or Python gap |
|---|---|---|---|
| P0 | Reuse one constructed basis/operator across many method calls | owned Rust model exists, but no stable handle registry or lifecycle contract | every C command currently rebuilds the model |
| P0 | `spin_basis_general`, arbitrary symmetry maps, and all 1D block combinations | generic `GeneralBasis` exists, but runtime-owned symmetry specifications are incomplete | no serializable symmetry-map schema |
| P0 | Complete `hamiltonian` object protocol | operator algebra exists | no persistent operator handle, NumPy/SciPy input, sparse export, transpose/adjoint/algebra command set |
| P1 | Tensor, photon, and general/user bases | native generic implementations exist with different state types | `PackedBasis` currently erases only `u128` packed bases |
| P1 | Python callables for `user_basis` and dynamic drives | Rust closures are supported | no callback/handle transport across the ABI |
| P1 | `Op`, `inplace_Op`, `Op_bra_ket`, and sector shifts | transition primitives exist | no batch array command or Python shape/dtype adapter |
| P1 | Evolution, Floquet, Lanczos, and exponential actions | native implementations exist | absent from the language-neutral command protocol |
| P1 | Entropy, partial trace, observables, and ensembles | native measurement implementations exist | absent from the language-neutral command protocol |
| P2 | QuSpin archive and internal helper fidelity | QMBED archive and generic utilities exist | exact QuSpin ZIP/layout, internal reshape, lattice helper, warning, and error semantics remain |

## Acceptance rule

Moving a file from `unsupported` to `passing` requires all of:

1. the copied upstream file remains byte-for-byte unchanged;
2. its original test entry points execute in CI;
3. new general Rust behavior has direct Rust tests;
4. the implementation does not branch on a model or test name;
5. the compatibility status remains exhaustive across all 73 files.
