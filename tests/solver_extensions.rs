use std::sync::Arc;

use approx::assert_abs_diff_eq;
use quspin::Complex64;
use quspin::operator::{ExpOp, LinearOperator, Operator};
use quspin::solve::{
    ExpmOptions, LanczosOptions, expm_multiply, ftlm_static_iteration, lanczos_full, lanczos_iter,
    linear_combination_qt, ltlm_static_iteration,
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
}
