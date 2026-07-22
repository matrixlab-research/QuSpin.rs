use std::f64::consts::{FRAC_2_PI, LN_2, PI};
use std::sync::Arc;

use approx::assert_abs_diff_eq;
use quspin::basis::{
    Basis, BosonBasis1D, SpinBasis1D, SpinfulFermionBasis1D, SpinlessFermionBasis1D, UserBasis,
};
use quspin::dynamics::{DriveStep, Floquet, SpectrumOptions, spectral_function};
use quspin::measure::{Subspace, subspace_fidelity};
use quspin::operator::{
    Coupling, LinearOperator, MatrixFormat, Operator, OperatorBuilder, OperatorTerm,
};
use quspin::solve::{EigshOptions, EvolutionOptions, SpectrumTarget, eigsh, evolve};
use quspin::workflow::LindbladGenerator;
use quspin::{Complex64, QuSpinError, Result};

fn c(value: f64) -> Complex64 {
    Complex64::new(value, 0.0)
}

fn periodic_heisenberg_terms(sites: usize) -> [OperatorTerm; 3] {
    let mut zz = Vec::with_capacity(sites);
    let mut forward = Vec::with_capacity(sites);
    let mut backward = Vec::with_capacity(sites);
    for site in 0..sites {
        let next = (site + 1) % sites;
        zz.push(Coupling::new(1.0, vec![site, next]));
        forward.push(Coupling::new(0.5, vec![site, next]));
        backward.push(Coupling::new(0.5, vec![site, next]));
    }
    [
        OperatorTerm::new("zz", zz).unwrap(),
        OperatorTerm::new("+-", forward).unwrap(),
        OperatorTerm::new("-+", backward).unwrap(),
    ]
}

struct DiagonalOperator {
    diagonal: Vec<f64>,
}

impl LinearOperator for DiagonalOperator {
    fn shape(&self) -> (usize, usize) {
        (self.diagonal.len(), self.diagonal.len())
    }

    fn format(&self) -> MatrixFormat {
        MatrixFormat::MatrixFree
    }

    fn apply(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        if input.len() != self.diagonal.len() || output.len() != self.diagonal.len() {
            return Err(QuSpinError::DimensionMismatch(
                "diagonal test operator shape mismatch".into(),
            ));
        }
        for ((result, input_value), diagonal) in output.iter_mut().zip(input).zip(&self.diagonal) {
            *result = *diagonal * *input_value;
        }
        Ok(())
    }
}

fn periodic_blockade(state: u128, sites: usize) -> bool {
    (0..sites).all(|site| {
        let next = (site + 1) % sites;
        !(state & (1_u128 << site) != 0 && state & (1_u128 << next) != 0)
    })
}

fn periodic_blockade_states(sites: usize) -> Vec<u128> {
    fn extend(
        site: usize,
        sites: usize,
        first_occupied: bool,
        previous_occupied: bool,
        state: u128,
        output: &mut Vec<u128>,
    ) {
        if site == sites {
            if !(first_occupied && previous_occupied) {
                output.push(state);
            }
            return;
        }
        extend(site + 1, sites, first_occupied, false, state, output);
        if !previous_occupied {
            extend(
                site + 1,
                sites,
                first_occupied || site == 0,
                true,
                state | (1_u128 << site),
                output,
            );
        }
    }

    let mut states = Vec::new();
    extend(0, sites, false, false, 0, &mut states);
    states.sort_unstable();
    states
}

#[test]
fn error_categories_are_structured() {
    let basis = SpinBasis1D::builder(4).up(2).build().unwrap();
    assert!(matches!(basis.state(6), Err(QuSpinError::StateNotInBasis)));
}

#[test]
fn built_in_basis_visible_anchors() {
    let spin = SpinBasis1D::builder(4).up(2).build().unwrap();
    assert_eq!(spin.len(), 6);
    let state = 0b0101;
    assert_eq!(spin.state(spin.index(state).unwrap()).unwrap(), state);

    let boson = BosonBasis1D::builder(3, 3).particles(2).build().unwrap();
    assert_eq!(boson.len(), 6);

    let spinless = SpinlessFermionBasis1D::builder(4)
        .particles(2)
        .build()
        .unwrap();
    assert_eq!(spinless.len(), 6);
    for index in 0..spinless.len() {
        assert_eq!(spinless.state(index).unwrap().count_ones(), 2);
    }

    let spinful = SpinfulFermionBasis1D::builder(3)
        .particles(1, 1)
        .build()
        .unwrap();
    assert_eq!(spinful.len(), 9);
}

#[test]
fn spin_translation_sector_uses_normalized_orbit_representatives() {
    let momentum_zero = SpinBasis1D::builder(4).up(2).momentum(0).build().unwrap();
    let momentum_one = SpinBasis1D::builder(4).up(2).momentum(1).build().unwrap();
    assert_eq!(momentum_zero.len(), 2);
    assert_eq!(momentum_one.len(), 1);
    assert_eq!(momentum_zero.momentum(), Some(0));

    let terms = periodic_heisenberg_terms(4);
    let hamiltonian = OperatorBuilder::on(&momentum_zero)
        .terms(terms.clone())
        .build(MatrixFormat::Csc)
        .unwrap();
    assert!(hamiltonian.is_hermitian(1.0e-12));
    let result = eigsh(
        &hamiltonian,
        EigshOptions {
            eigenpairs: 1,
            target: SpectrumTarget::SmallestAlgebraic,
            krylov_dimension: None,
            tolerance: 1.0e-12,
            max_iterations: 100,
            seed: 5,
        },
    )
    .unwrap();
    assert_abs_diff_eq!(result.eigenvalues[0], -2.0, epsilon = 1.0e-12);

    let nonzero_momentum = OperatorBuilder::on(&momentum_one)
        .terms(terms)
        .build(MatrixFormat::Csc)
        .unwrap();
    assert!(nonzero_momentum.is_hermitian(1.0e-12));
    assert_abs_diff_eq!(nonzero_momentum.to_dense()[0].re, 0.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(nonzero_momentum.to_dense()[0].im, 0.0, epsilon = 1.0e-12);
}

#[test]
#[ignore = "paper-scale workflow; exercised in release mode"]
fn paper_scale_translation_xxz_sector_stays_sparse() {
    let basis = SpinBasis1D::builder(18).up(9).momentum(0).build().unwrap();
    assert_eq!(basis.len(), 2_704);
    let hamiltonian = OperatorBuilder::on(&basis)
        .terms(periodic_heisenberg_terms(18))
        .build(MatrixFormat::Csc)
        .unwrap();
    assert_eq!(hamiltonian.shape(), (2_704, 2_704));
    assert!(hamiltonian.nnz() < 60_000);
    let result = eigsh(
        &hamiltonian,
        EigshOptions {
            eigenpairs: 4,
            target: SpectrumTarget::SmallestAlgebraic,
            krylov_dimension: Some(96),
            tolerance: 1.0e-8,
            max_iterations: 128,
            seed: 31,
        },
    )
    .unwrap();
    assert!(result.residuals.iter().all(|residual| *residual < 2.0e-7));
}

#[test]
#[ignore = "paper-scale workflow; exercised in release mode"]
fn paper_scale_xxz_lanczos_quench_preserves_norm() {
    let sites = 16;
    let basis = SpinBasis1D::builder(sites).up(8).build().unwrap();
    assert_eq!(basis.len(), 12_870);
    let bonds = 0..(sites - 1);
    let hamiltonian = OperatorBuilder::on(&basis)
        .terms([
            OperatorTerm::new(
                "+-",
                bonds
                    .clone()
                    .map(|site| Coupling::new(0.5, vec![site, site + 1])),
            )
            .unwrap(),
            OperatorTerm::new(
                "-+",
                bonds
                    .clone()
                    .map(|site| Coupling::new(0.5, vec![site, site + 1])),
            )
            .unwrap(),
            OperatorTerm::new(
                "zz",
                bonds.map(|site| Coupling::new(0.8, vec![site, site + 1])),
            )
            .unwrap(),
        ])
        .build(MatrixFormat::Csc)
        .unwrap();
    let neel = (0..sites)
        .step_by(2)
        .fold(0_u128, |state, site| state | (1_u128 << site));
    let mut initial = vec![c(0.0); basis.len()];
    initial[basis.index(neel).unwrap()] = c(1.0);
    let trajectory = evolve(
        &hamiltonian,
        &initial,
        EvolutionOptions {
            times: vec![0.7],
            krylov_dimension: 80,
            tolerance: 1.0e-10,
            max_substeps: 100,
            hamiltonian: true,
        },
    )
    .unwrap();
    let norm = trajectory.states[0]
        .iter()
        .map(Complex64::norm_sqr)
        .sum::<f64>()
        .sqrt();
    assert!((norm - 1.0).abs() < 2.0e-9);
}

#[test]
#[ignore = "paper-scale workflow; exercised in release mode"]
fn paper_scale_pxp_revival_uses_the_universal_user_basis_path() {
    let sites = 24;
    let basis = UserBasis::builder(sites)
        .states(periodic_blockade_states(sites))
        .operator('x', |state, site| {
            Ok(Some((state ^ (1_u128 << site), Complex64::new(1.0, 0.0))))
        })
        .build()
        .unwrap();
    assert_eq!(basis.len(), 103_682);
    let hamiltonian = OperatorBuilder::on(&basis)
        .term(
            OperatorTerm::new("x", (0..sites).map(|site| Coupling::new(1.0, vec![site]))).unwrap(),
        )
        .build(MatrixFormat::Csc)
        .unwrap();
    let neel = (0..sites)
        .step_by(2)
        .fold(0_u128, |state, site| state | (1_u128 << site));
    let mut initial = vec![c(0.0); basis.len()];
    initial[basis.index(neel).unwrap()] = c(1.0);
    let trajectory = evolve(
        &hamiltonian,
        &initial,
        EvolutionOptions {
            times: vec![0.0, 2.4, 4.8, 7.2, 9.6],
            krylov_dimension: 100,
            tolerance: 1.0e-9,
            max_substeps: 100,
            hamiltonian: true,
        },
    )
    .unwrap();
    let mut fidelities = Vec::new();
    for state in &trajectory.states {
        let norm = state.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 5.0e-8, "evolved norm was {norm}");
        fidelities.push(state[basis.index(neel).unwrap()].norm_sqr());
    }
    assert!(fidelities[2] > fidelities[1]);
}

#[test]
#[ignore = "paper-scale workflow; exercised in release mode"]
fn paper_scale_bose_hubbard_mott_quench_reuses_one_krylov_projection() {
    let sites = 11;
    let basis = BosonBasis1D::builder(sites, 3)
        .particles(sites)
        .build()
        .unwrap();
    assert_eq!(basis.len(), 25_653);
    let bonds = 0..(sites - 1);
    let hamiltonian = OperatorBuilder::on(&basis)
        .terms([
            OperatorTerm::new(
                "+-",
                bonds
                    .clone()
                    .map(|site| Coupling::new(-0.1, vec![site, site + 1])),
            )
            .unwrap(),
            OperatorTerm::new(
                "-+",
                bonds
                    .clone()
                    .map(|site| Coupling::new(-0.1, vec![site, site + 1])),
            )
            .unwrap(),
            OperatorTerm::new(
                "nn",
                (0..sites).map(|site| Coupling::new(0.5, vec![site, site])),
            )
            .unwrap(),
            OperatorTerm::new("n", (0..sites).map(|site| Coupling::new(-0.5, vec![site]))).unwrap(),
        ])
        .build(MatrixFormat::Csc)
        .unwrap();
    let mott = (0..sites).map(|site| 3_u128.pow(site as u32)).sum();
    let mut initial = vec![c(0.0); basis.len()];
    initial[basis.index(mott).unwrap()] = c(1.0);
    let trajectory = evolve(
        &hamiltonian,
        &initial,
        EvolutionOptions {
            times: vec![0.0, 25.0, 50.0, 100.0, 200.0],
            krylov_dimension: 100,
            tolerance: 1.0e-9,
            max_substeps: 1_000,
            hamiltonian: true,
        },
    )
    .unwrap();
    let mut returns = Vec::new();
    for state in &trajectory.states {
        let norm = state.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 5.0e-8);
        returns.push(state[basis.index(mott).unwrap()].norm_sqr());
    }
    assert!(returns[1..].iter().copied().fold(1.0_f64, f64::min) < 0.99);
}

#[test]
#[ignore = "paper-scale workflow; exercised in release mode"]
fn paper_scale_spinful_hubbard_low_energy_residuals() {
    let sites = 8;
    let basis = SpinfulFermionBasis1D::builder(sites)
        .particles(4, 4)
        .build()
        .unwrap();
    assert_eq!(basis.len(), 4_900);
    let bonds = 0..(sites - 1);
    let hamiltonian = OperatorBuilder::on(&basis)
        .terms([
            OperatorTerm::new(
                "+-|",
                bonds
                    .clone()
                    .map(|site| Coupling::new(-1.0, vec![site, site + 1])),
            )
            .unwrap(),
            OperatorTerm::new(
                "-+|",
                bonds
                    .clone()
                    .map(|site| Coupling::new(1.0, vec![site, site + 1])),
            )
            .unwrap(),
            OperatorTerm::new(
                "|+-",
                bonds
                    .clone()
                    .map(|site| Coupling::new(-1.0, vec![site, site + 1])),
            )
            .unwrap(),
            OperatorTerm::new(
                "|-+",
                bonds.map(|site| Coupling::new(1.0, vec![site, site + 1])),
            )
            .unwrap(),
            OperatorTerm::new(
                "n|n",
                (0..sites).map(|site| Coupling::new(4.0, vec![site, site])),
            )
            .unwrap(),
        ])
        .build(MatrixFormat::Csc)
        .unwrap();
    let result = eigsh(
        &hamiltonian,
        EigshOptions {
            eigenpairs: 6,
            target: SpectrumTarget::SmallestAlgebraic,
            krylov_dimension: Some(160),
            tolerance: 1.0e-9,
            max_iterations: 192,
            seed: 37,
        },
    )
    .unwrap();
    assert!(result.residuals.iter().all(|residual| *residual < 2.0e-7));
}

#[test]
#[ignore = "paper-scale workflow; exercised in release mode"]
fn paper_scale_interacting_ssh_low_energy_residuals() {
    let sites = 16;
    let basis = SpinlessFermionBasis1D::builder(sites)
        .particles(8)
        .build()
        .unwrap();
    assert_eq!(basis.len(), 12_870);
    let hopping = |site: usize| if site % 2 == 0 { 0.6 } else { 1.0 };
    let bonds = 0..(sites - 1);
    let hamiltonian = OperatorBuilder::on(&basis)
        .terms([
            OperatorTerm::new(
                "+-",
                bonds
                    .clone()
                    .map(|site| Coupling::new(-hopping(site), vec![site, site + 1])),
            )
            .unwrap(),
            OperatorTerm::new(
                "-+",
                bonds
                    .clone()
                    .map(|site| Coupling::new(hopping(site), vec![site, site + 1])),
            )
            .unwrap(),
            OperatorTerm::new(
                "nn",
                bonds.map(|site| Coupling::new(2.0, vec![site, site + 1])),
            )
            .unwrap(),
        ])
        .build(MatrixFormat::Csc)
        .unwrap();
    let result = eigsh(
        &hamiltonian,
        EigshOptions {
            eigenpairs: 6,
            target: SpectrumTarget::SmallestAlgebraic,
            krylov_dimension: Some(160),
            tolerance: 1.0e-9,
            max_iterations: 192,
            seed: 41,
        },
    )
    .unwrap();
    assert!(result.residuals.iter().all(|residual| *residual < 2.0e-7));
}

#[test]
#[ignore = "paper-scale workflow; exercised in release mode"]
fn paper_scale_tfim_tracks_degenerate_subspaces() {
    let sites = 16;
    let basis = SpinBasis1D::builder(sites).pauli(true).build().unwrap();
    assert_eq!(basis.len(), 65_536);
    let mut subspaces = Vec::new();
    for (field_index, field) in [0.8, 0.9, 1.0, 1.1, 1.2].into_iter().enumerate() {
        let hamiltonian = OperatorBuilder::on(&basis)
            .terms([
                OperatorTerm::new(
                    "zz",
                    (0..sites).map(|site| Coupling::new(-1.0, vec![site, (site + 1) % sites])),
                )
                .unwrap(),
                OperatorTerm::new(
                    "x",
                    (0..sites).map(|site| Coupling::new(-field, vec![site])),
                )
                .unwrap(),
            ])
            .build(MatrixFormat::Csc)
            .unwrap();
        let result = eigsh(
            &hamiltonian,
            EigshOptions {
                eigenpairs: 2,
                target: SpectrumTarget::SmallestAlgebraic,
                krylov_dimension: Some(100),
                tolerance: 1.0e-9,
                max_iterations: 128,
                seed: 43 + field_index as u64,
            },
        )
        .unwrap();
        assert!(result.residuals.iter().all(|residual| *residual < 3.0e-7));
        subspaces.push(
            Subspace::from_columns(
                basis.len(),
                2,
                result.eigenvectors.into_iter().flatten().collect(),
            )
            .unwrap(),
        );
    }
    let fidelities: Vec<_> = subspaces
        .windows(2)
        .map(|pair| subspace_fidelity(&pair[0], &pair[1]).unwrap())
        .collect();
    assert!(
        fidelities
            .iter()
            .all(|value| *value > 0.0 && *value <= 1.0 + 1.0e-12)
    );
    let minimum = fidelities
        .iter()
        .enumerate()
        .min_by(|left, right| left.1.total_cmp(right.1))
        .unwrap()
        .0;
    assert!((1..=2).contains(&minimum));
}

#[test]
#[ignore = "paper-scale workflow; exercised in release mode"]
fn paper_scale_mbl_uses_reusable_sparse_shift_invert() {
    let sites = 14;
    let basis = SpinBasis1D::builder(sites).up(7).build().unwrap();
    assert_eq!(basis.len(), 3_432);
    let fields = [
        2.13, -1.77, 0.31, 3.24, -2.63, 0.82, 1.46, -3.17, 2.71, -0.54, 1.09, -2.28, 0.67, 2.94,
    ];
    let mut terms = periodic_heisenberg_terms(sites).to_vec();
    terms.push(
        OperatorTerm::new(
            "z",
            fields
                .into_iter()
                .enumerate()
                .map(|(site, field)| Coupling::new(field, vec![site])),
        )
        .unwrap(),
    );
    let hamiltonian = OperatorBuilder::on(&basis)
        .terms(terms)
        .build(MatrixFormat::Csc)
        .unwrap();
    let result = eigsh(
        &hamiltonian,
        EigshOptions {
            eigenpairs: 6,
            target: SpectrumTarget::Shift(0.0),
            krylov_dimension: Some(32),
            tolerance: 1.0e-9,
            max_iterations: 5_000,
            seed: 47,
        },
    )
    .unwrap();
    let combined_residual = result
        .residuals
        .iter()
        .map(|residual| residual * residual)
        .sum::<f64>()
        .sqrt();
    assert!(combined_residual < 2.0e-7);
}

#[test]
fn user_basis_visible_anchor_uses_universal_contract() -> Result<()> {
    let basis = UserBasis::<u128>::builder(4)
        .state_filter(|state| periodic_blockade(state, 4))?
        .operator('x', |state, site| {
            Ok(Some((state ^ (1_u128 << site), c(1.0))))
        })
        .operator('z', |state, site| {
            let value = if state & (1_u128 << site) == 0 {
                -1.0
            } else {
                1.0
            };
            Ok(Some((state, c(value))))
        })
        .build()?;
    assert_eq!(basis.len(), 7);
    for index in 0..basis.len() {
        let state = basis.state(index)?;
        assert_eq!(basis.index(state)?, index);
    }
    Ok(())
}

#[test]
fn terms_couplings_and_formats_are_explicit() {
    let term = OperatorTerm::new(
        "zz",
        [
            Coupling::new(1.0, vec![0, 1]),
            Coupling::new(-0.5, vec![1, 2]),
        ],
    )
    .unwrap();
    assert_eq!(term.operator(), "zz");
    assert_eq!(term.couplings().len(), 2);
    assert_eq!(term.couplings()[0].sites, [0, 1]);
    assert_eq!(
        [
            MatrixFormat::Dense,
            MatrixFormat::Csc,
            MatrixFormat::Csr,
            MatrixFormat::Dia,
            MatrixFormat::MatrixFree,
        ]
        .len(),
        5
    );
}

#[test]
fn linear_operator_is_rectangular_capable() {
    let operator =
        Operator::from_dense(2, 3, vec![c(1.0), c(0.0), c(2.0), c(0.0), c(-1.0), c(1.0)]).unwrap();
    let mut output = vec![c(0.0); 2];
    operator
        .apply(&[c(1.0), c(2.0), c(3.0)], &mut output)
        .unwrap();
    assert_eq!(operator.shape(), (2, 3));
    assert_abs_diff_eq!(output[0].re, 7.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(output[1].re, 1.0, epsilon = 1.0e-12);
}

fn heisenberg_dimer(format: MatrixFormat) -> Operator {
    let basis = SpinBasis1D::builder(2).pauli(false).build().unwrap();
    let terms = [
        OperatorTerm::new("zz", [Coupling::new(1.0, vec![0, 1])]).unwrap(),
        OperatorTerm::new("+-", [Coupling::new(0.5, vec![0, 1])]).unwrap(),
        OperatorTerm::new("-+", [Coupling::new(0.5, vec![0, 1])]).unwrap(),
    ];
    OperatorBuilder::on(&basis)
        .terms(terms)
        .build(format)
        .unwrap()
}

#[test]
fn universal_builder_and_eigsh_match_the_dimer_anchor() {
    let operator = heisenberg_dimer(MatrixFormat::Csc);
    assert_eq!(operator.shape(), (4, 4));
    assert!(operator.is_hermitian(1.0e-12));
    let result = eigsh(
        &operator,
        EigshOptions {
            eigenpairs: 2,
            target: SpectrumTarget::SmallestAlgebraic,
            krylov_dimension: None,
            tolerance: 1.0e-12,
            max_iterations: 100,
            seed: 7,
        },
    )
    .unwrap();
    assert_abs_diff_eq!(result.eigenvalues[0], -0.75, epsilon = 1.0e-12);
    assert_abs_diff_eq!(result.eigenvalues[1], 0.25, epsilon = 1.0e-12);
    assert!(result.residuals.iter().all(|residual| *residual < 1.0e-12));
}

#[test]
fn all_materialization_formats_preserve_sparse_action() {
    let input = [
        Complex64::new(1.0, -0.5),
        Complex64::new(-2.0, 0.25),
        Complex64::new(0.75, 1.0),
        Complex64::new(0.5, -1.5),
    ];
    let dense = heisenberg_dimer(MatrixFormat::Dense);
    let expected_dense = dense.to_dense();
    let mut expected = vec![c(0.0); 4];
    dense.apply(&input, &mut expected).unwrap();

    for format in [
        MatrixFormat::Csc,
        MatrixFormat::Csr,
        MatrixFormat::Dia,
        MatrixFormat::MatrixFree,
    ] {
        let operator = heisenberg_dimer(format);
        let mut actual = vec![c(0.0); 4];
        operator.apply(&input, &mut actual).unwrap();
        assert_eq!(operator.format(), format);
        assert_eq!(operator.nnz(), 6);
        assert_eq!(operator.to_dense(), expected_dense);
        for (actual, expected) in actual.iter().zip(&expected) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }
}

#[test]
fn lanczos_backend_avoids_dense_materialization_for_large_operators() {
    let operator = DiagonalOperator {
        diagonal: (0..256).map(|value| value as f64).collect(),
    };
    let result = eigsh(
        &operator,
        EigshOptions {
            eigenpairs: 3,
            target: SpectrumTarget::SmallestAlgebraic,
            krylov_dimension: Some(160),
            tolerance: 1.0e-8,
            max_iterations: 192,
            seed: 17,
        },
    )
    .unwrap();
    assert_abs_diff_eq!(result.eigenvalues[0], 0.0, epsilon = 1.0e-7);
    assert_abs_diff_eq!(result.eigenvalues[1], 1.0, epsilon = 1.0e-7);
    assert_abs_diff_eq!(result.eigenvalues[2], 2.0, epsilon = 1.0e-7);
    assert!(result.residuals.iter().all(|residual| *residual < 1.0e-7));
    assert_eq!(result.iterations, 160);
}

#[test]
fn sparse_shift_invert_finds_interior_eigenpairs() {
    let operator = Operator::from_triplets(
        256,
        256,
        (-128..128)
            .enumerate()
            .map(|(index, value)| (index, index, Complex64::new(f64::from(value), 0.0))),
        MatrixFormat::Csc,
    )
    .unwrap();
    let result = eigsh(
        &operator,
        EigshOptions {
            eigenpairs: 2,
            target: SpectrumTarget::Shift(0.3),
            krylov_dimension: Some(24),
            tolerance: 1.0e-8,
            max_iterations: 512,
            seed: 23,
        },
    )
    .unwrap();
    assert_abs_diff_eq!(result.eigenvalues[0], 0.0, epsilon = 1.0e-7);
    assert_abs_diff_eq!(result.eigenvalues[1], 1.0, epsilon = 1.0e-7);
    assert!(result.residuals.iter().all(|residual| *residual < 1.0e-7));
}

#[test]
fn cross_sector_builder_has_target_by_source_shape() {
    let source = SpinlessFermionBasis1D::builder(4)
        .particles(1)
        .build()
        .unwrap();
    let target = SpinlessFermionBasis1D::builder(4)
        .particles(2)
        .build()
        .unwrap();
    let probe = OperatorBuilder::between(&source, &target)
        .term(OperatorTerm::new("+", [Coupling::new(1.0, vec![2])]).unwrap())
        .build(MatrixFormat::Csc)
        .unwrap();
    assert_eq!(probe.shape(), (6, 4));
}

#[test]
fn evolution_and_floquet_match_diagonal_visible_anchors() {
    let diagonal =
        Arc::new(Operator::from_dense(2, 2, vec![c(0.0), c(0.0), c(0.0), c(2.0)]).unwrap());
    let initial = vec![c(1.0 / 2.0_f64.sqrt()), c(1.0 / 2.0_f64.sqrt())];
    let options = EvolutionOptions {
        times: vec![0.0, PI / 2.0],
        krylov_dimension: 64,
        tolerance: 1.0e-12,
        max_substeps: 100,
        hamiltonian: true,
    };
    let trajectory = evolve(diagonal.as_ref(), &initial, options).unwrap();
    assert_abs_diff_eq!(trajectory.states[0][1].re, initial[1].re, epsilon = 1.0e-12);
    assert_abs_diff_eq!(
        trajectory.states[1][1].re,
        -initial[1].re,
        epsilon = 1.0e-10
    );
    assert_abs_diff_eq!(trajectory.states[1][1].im, 0.0, epsilon = 1.0e-10);

    let zero = Arc::new(Operator::from_dense(2, 2, vec![c(0.0); 4]).unwrap());
    let floquet = Floquet::new([
        DriveStep::new(diagonal, PI / 2.0).unwrap(),
        DriveStep::new(zero, 3.0).unwrap(),
    ])
    .unwrap();
    let mut output = vec![c(0.0); 2];
    floquet.apply_period(&initial, &mut output).unwrap();
    assert_abs_diff_eq!(output[1].re, -initial[1].re, epsilon = 1.0e-10);
}

#[test]
fn spectrum_visible_anchor_is_one_lorentzian_pole() {
    let hamiltonian = Operator::from_dense(1, 1, vec![c(1.0)]).unwrap();
    let probe = Operator::from_dense(1, 1, vec![c(1.0)]).unwrap();
    let spectrum = spectral_function(
        &hamiltonian,
        &[c(1.0)],
        &probe,
        SpectrumOptions {
            frequencies: vec![1.0],
            reference_energy: 0.0,
            broadening: 0.5,
            krylov_dimension: 8,
            tolerance: 1.0e-12,
        },
    )
    .unwrap();
    assert_abs_diff_eq!(spectrum[0], FRAC_2_PI, epsilon = 1.0e-12);
}

#[test]
fn subspace_fidelity_is_rotation_invariant() {
    let scale = 1.0 / 2.0_f64.sqrt();
    let left =
        Subspace::from_columns(3, 2, vec![c(1.0), c(0.0), c(0.0), c(0.0), c(1.0), c(0.0)]).unwrap();
    let right = Subspace::from_columns(
        3,
        2,
        vec![c(scale), c(scale), c(0.0), c(scale), c(-scale), c(0.0)],
    )
    .unwrap();
    assert_abs_diff_eq!(
        subspace_fidelity(&left, &right).unwrap(),
        1.0,
        epsilon = 1.0e-12
    );
}

#[test]
fn lindblad_amplitude_damping_preserves_trace() {
    let zero = Arc::new(Operator::from_dense(2, 2, vec![c(0.0); 4]).unwrap());
    let lowering =
        Arc::new(Operator::from_dense(2, 2, vec![c(0.0), c(1.0), c(0.0), c(0.0)]).unwrap());
    let generator = LindbladGenerator::new(zero, vec![lowering]).unwrap();
    let initial_density_column_major = vec![c(0.0), c(0.0), c(0.0), c(1.0)];
    let trajectory = evolve(
        &generator,
        &initial_density_column_major,
        EvolutionOptions {
            times: vec![LN_2],
            krylov_dimension: 64,
            tolerance: 1.0e-12,
            max_substeps: 100,
            hamiltonian: false,
        },
    )
    .unwrap();
    let density = &trajectory.states[0];
    assert_abs_diff_eq!(density[0].re + density[3].re, 1.0, epsilon = 1.0e-10);
    assert_abs_diff_eq!(density[3].re, 0.5, epsilon = 1.0e-10);
}
