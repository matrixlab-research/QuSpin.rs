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
fn shift_invert_finds_interior_eigenpairs_matrix_free() {
    let operator = DiagonalOperator {
        diagonal: (-128..128).map(f64::from).collect(),
    };
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
