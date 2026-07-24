use std::sync::Arc;

use approx::assert_abs_diff_eq;
use qmbed::Complex64;
use qmbed::operator::{ExpOp, LinearOperator, MatrixFormat, Operator};
use qmbed::solve::{
    EigshOptions, ExpmMultiplyParallel, ExpmOptions, LanczosOptions, ShiftInvertPlan,
    SpectrumTarget, eigsh, expm_multiply, ftlm_observable_iteration, ftlm_static_iteration,
    lanczos_full, lanczos_iter, linear_combination_qt, ltlm_observable_iteration,
    ltlm_static_iteration,
};

fn inner(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left.conj() * *right)
        .sum()
}

fn diagonal(values: &[f64]) -> Operator {
    let dimension = values.len();
    let mut dense = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    for (index, value) in values.iter().enumerate() {
        dense[index * dimension + index] = Complex64::new(*value, 0.0);
    }
    Operator::from_dense(dimension, dimension, dense).unwrap()
}

#[test]
fn real_operator_capability_drives_large_sparse_eigsh() {
    let dimension = 160;
    let operator = Operator::from_triplets(
        dimension,
        dimension,
        [
            (0, 0, Complex64::new(-2.0, 0.0)),
            (1, 1, Complex64::new(-1.0, 0.0)),
        ],
        MatrixFormat::Csc,
    )
    .unwrap();
    assert!(operator.is_real());
    let mut real_output = vec![0.0; dimension];
    operator
        .apply_real(&vec![1.0; dimension], &mut real_output)
        .unwrap();
    assert_eq!(&real_output[..3], &[-2.0, -1.0, 0.0]);

    let result = eigsh(
        &operator,
        EigshOptions {
            eigenpairs: 2,
            target: SpectrumTarget::SmallestAlgebraic,
            krylov_dimension: Some(8),
            tolerance: 1.0e-12,
            max_iterations: 32,
            seed: 7,
        },
    )
    .unwrap();
    assert_abs_diff_eq!(result.eigenvalues[0], -2.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(result.eigenvalues[1], -1.0, epsilon = 1.0e-12);
    assert!(result.residuals.iter().all(|residual| *residual < 1.0e-12));

    let complex =
        Operator::from_triplets(1, 1, [(0, 0, Complex64::new(0.0, 1.0))], MatrixFormat::Csc)
            .unwrap();
    assert!(!complex.is_real());
    assert!(complex.apply_real(&[1.0], &mut [0.0]).is_err());
}

#[test]
fn public_lanczos_full_and_iterator_return_the_same_tridiagonalization() {
    let operator = diagonal(&[-2.0, 0.5, 3.0]);
    let initial = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(-1.0, 0.0),
    ];
    let options = LanczosOptions {
        krylov_dimension: 3,
        tolerance: 1.0e-13,
    };
    let full = lanczos_full(&operator, &initial, options.clone()).unwrap();
    let streamed: Vec<_> = lanczos_iter(&operator, &initial, options)
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(full.basis.len(), 3);
    assert_eq!(streamed.len(), 3);
    for (index, vector) in full.basis.iter().enumerate() {
        for (other_index, other) in full.basis.iter().enumerate() {
            let expected = if index == other_index { 1.0 } else { 0.0 };
            assert_abs_diff_eq!(inner(vector, other).re, expected, epsilon = 1.0e-11);
        }
        for (actual, expected) in vector.iter().zip(&streamed[index].vector) {
            assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
            assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
        }
    }
}

#[test]
fn expop_supports_unitary_and_general_complex_exponents() {
    let operator = Arc::new(diagonal(&[-1.0, 2.0]));
    let initial = vec![
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ];
    let unitary = ExpOp::new(
        operator.clone(),
        Complex64::new(0.0, -0.4),
        16,
        1.0e-12,
        100,
    )
    .unwrap();
    let mut output = vec![Complex64::new(0.0, 0.0); 2];
    unitary.apply(&initial, &mut output).unwrap();
    let expected = [
        initial[0] * Complex64::new(0.0, 0.4).exp(),
        initial[1] * Complex64::new(0.0, -0.8).exp(),
    ];
    for (actual, expected) in output.iter().zip(expected) {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-11);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-11);
    }

    let thermal = ExpOp::new(operator, Complex64::new(-0.5, 0.0), 32, 1.0e-13, 100).unwrap();
    thermal.apply(&initial, &mut output).unwrap();
    assert_abs_diff_eq!(
        output[0].re,
        initial[0].re * 0.5_f64.exp(),
        epsilon = 1.0e-11
    );
    assert_abs_diff_eq!(
        output[1].re,
        initial[1].re * (-1.0_f64).exp(),
        epsilon = 1.0e-11
    );
}

#[test]
fn expm_multiply_reuses_the_existing_trajectory_contract() {
    let operator = diagonal(&[-1.0, 1.0]);
    let initial = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    let trajectory = expm_multiply(
        &operator,
        &initial,
        ExpmOptions {
            times: vec![0.0, 0.25, 0.5],
            krylov_dimension: 8,
            tolerance: 1.0e-12,
            max_substeps: 100,
            hamiltonian: true,
        },
    )
    .unwrap();
    assert_eq!(trajectory.states.len(), 3);
    assert_abs_diff_eq!(trajectory.states[2][0].norm(), 1.0, epsilon = 1.0e-12);
}

#[test]
fn thermal_lanczos_matches_an_exact_two_level_trace() {
    let operator = diagonal(&[-1.0, 2.0]);
    let initial = vec![
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
    ];
    let options = LanczosOptions {
        krylov_dimension: 2,
        tolerance: 1.0e-13,
    };
    let beta = 0.7;
    let ftlm = ftlm_static_iteration(&operator, &initial, &[0.0, beta], options.clone()).unwrap();
    let ltlm = ltlm_static_iteration(&operator, &initial, &[0.0, beta], options).unwrap();
    let partition = beta.exp() + (-2.0 * beta).exp();
    let energy = (-beta.exp() + 2.0 * (-2.0 * beta).exp()) / partition;
    assert_abs_diff_eq!(ftlm.log_partition[0], 2.0_f64.ln(), epsilon = 1.0e-12);
    assert_abs_diff_eq!(ftlm.log_partition[1], partition.ln(), epsilon = 1.0e-12);
    assert_abs_diff_eq!(ftlm.mean_energy[1], energy, epsilon = 1.0e-12);
    assert_eq!(ftlm.log_partition, ltlm.log_partition);

    let combined = linear_combination_qt(
        &[
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ],
        &[Complex64::new(2.0, 0.0), Complex64::new(0.0, -1.0)],
    )
    .unwrap();
    assert_eq!(
        combined,
        vec![Complex64::new(2.0, 0.0), Complex64::new(0.0, -1.0)]
    );

    let identity = diagonal(&[1.0, 1.0]);
    let observables: Vec<(String, &dyn LinearOperator)> =
        vec![("H".to_string(), &operator), ("I".to_string(), &identity)];
    let observable_options = LanczosOptions {
        krylov_dimension: 2,
        tolerance: 1.0e-13,
    };
    let ftlm_observables = ftlm_observable_iteration(
        &operator,
        &initial,
        &observables,
        &[beta],
        observable_options.clone(),
    )
    .unwrap();
    let ltlm_observables = ltlm_observable_iteration(
        &operator,
        &initial,
        &observables,
        &[beta],
        observable_options,
    )
    .unwrap();
    assert_abs_diff_eq!(
        ftlm_observables.values["I"][0].re,
        ftlm_observables.identity[0],
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        ftlm_observables.values["H"][0].re / ftlm_observables.identity[0],
        energy,
        epsilon = 1.0e-12
    );
    assert_abs_diff_eq!(
        ltlm_observables.values["H"][0].re / ltlm_observables.identity[0],
        energy,
        epsilon = 1.0e-12
    );
}

#[test]
fn reusable_exponential_plan_supports_batches_and_coefficient_updates() {
    let generator = Arc::new(
        Operator::from_dense(
            2,
            2,
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap(),
    );
    let mut plan =
        ExpmMultiplyParallel::new(generator, Complex64::new(0.25, 0.0), 32, 1.0e-14, 100).unwrap();
    let batch = plan
        .apply_batch(&[
            vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
        ])
        .unwrap();
    assert_abs_diff_eq!(batch[0][0].re, 0.25_f64.cos(), epsilon = 1.0e-13);
    assert_abs_diff_eq!(batch[0][1].re, 0.25_f64.sin(), epsilon = 1.0e-13);
    assert_abs_diff_eq!(batch[1][0].re, -0.25_f64.sin(), epsilon = 1.0e-13);
    assert_abs_diff_eq!(batch[1][1].re, 0.25_f64.cos(), epsilon = 1.0e-13);
    plan.set_coefficient(Complex64::new(0.0, -0.5)).unwrap();
    let mut state = vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    plan.apply_in_place(&mut state).unwrap();
    assert_abs_diff_eq!(state[0].re, 0.5_f64.cosh(), epsilon = 1.0e-12);
    assert_abs_diff_eq!(state[1].im, -0.5_f64.sinh(), epsilon = 1.0e-12);
}

#[test]
fn reusable_shift_invert_plan_caches_sparse_factorization() {
    let operator = Arc::new(
        Operator::from_triplets(
            3,
            3,
            [
                (0, 0, Complex64::new(1.0, 0.0)),
                (1, 1, Complex64::new(2.0, 0.0)),
                (2, 2, Complex64::new(4.0, 0.0)),
            ],
            MatrixFormat::Csc,
        )
        .unwrap(),
    );
    let plan = ShiftInvertPlan::new(operator, 0.5, 1.0e-12, 100).unwrap();
    assert!(plan.is_factorized());
    let mut output = vec![Complex64::new(0.0, 0.0); 3];
    plan.solve(&[Complex64::new(1.0, 0.0); 3], &mut output)
        .unwrap();
    assert_abs_diff_eq!(output[0].re, 2.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(output[1].re, 2.0 / 3.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(output[2].re, 2.0 / 7.0, epsilon = 1.0e-12);
}
