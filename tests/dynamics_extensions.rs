use std::sync::Arc;

use approx::assert_abs_diff_eq;
use qmbed::dynamics::{
    CallableDriveStep, DriveStep, Floquet, FloquetTimeVector, dynamical_correlator,
};
use qmbed::operator::{Dynamic, DynamicComponent, Hamiltonian, MatrixFormat, Operator};
use qmbed::solve::EvolutionOptions;
use qmbed::{Complex64, QmbedError};

fn diagonal(values: &[f64]) -> Operator {
    let dimension = values.len();
    let mut dense = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    for (index, value) in values.iter().enumerate() {
        dense[index * dimension + index] = Complex64::new(*value, 0.0);
    }
    Operator::from_dense(dimension, dimension, dense).unwrap()
}

#[test]
fn floquet_builds_unitary_quasienergies_and_effective_hamiltonian() {
    let hamiltonian = diagonal(&[-1.0, 1.0]);
    let floquet =
        Floquet::new([DriveStep::new(Arc::new(hamiltonian.clone()), 0.7).unwrap()]).unwrap();
    assert_abs_diff_eq!(floquet.period(), 0.7, epsilon = 1.0e-15);
    let unitary = floquet.full_unitary(MatrixFormat::Csc).unwrap();
    let values = unitary.to_dense();
    assert_abs_diff_eq!(values[0].re, 0.7_f64.cos(), epsilon = 1.0e-12);
    assert_abs_diff_eq!(values[0].im, 0.7_f64.sin(), epsilon = 1.0e-12);
    assert_abs_diff_eq!(values[3].im, -0.7_f64.sin(), epsilon = 1.0e-12);

    let eigensystem = floquet.eigensystem().unwrap();
    assert_abs_diff_eq!(eigensystem.quasienergies[0], -1.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(eigensystem.quasienergies[1], 1.0, epsilon = 1.0e-12);
    assert!(
        eigensystem
            .residuals
            .iter()
            .all(|residual| *residual < 1.0e-12)
    );
    let effective = floquet.effective_hamiltonian(MatrixFormat::Dense).unwrap();
    for (actual, expected) in effective.to_dense().iter().zip(hamiltonian.to_dense()) {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
    }
}

#[test]
fn callable_floquet_drive_integrates_within_the_period() {
    let zero = diagonal(&[0.0, 0.0]);
    let driven = Hamiltonian::<Dynamic>::new(
        zero,
        vec![DynamicComponent::new(diagonal(&[-1.0, 1.0]), |time| {
            Complex64::new(time, 0.0)
        })],
    )
    .unwrap();
    let floquet =
        Floquet::from_callable([CallableDriveStep::new(Arc::new(driven), 1.0).unwrap()]).unwrap();
    let unitary = floquet
        .full_unitary(MatrixFormat::Dense)
        .unwrap()
        .to_dense();
    assert_abs_diff_eq!(unitary[0].re, 0.5_f64.cos(), epsilon = 2.0e-9);
    assert_abs_diff_eq!(unitary[0].im, 0.5_f64.sin(), epsilon = 2.0e-9);
    assert_abs_diff_eq!(unitary[3].im, -0.5_f64.sin(), epsilon = 2.0e-9);
}

#[test]
fn floquet_time_vector_has_exact_cycle_coordinates() {
    let times = FloquetTimeVector::new(2.0, 2, 4, true).unwrap();
    assert_eq!(times.times().len(), 9);
    assert_abs_diff_eq!(times.times()[8], 4.0, epsilon = 1.0e-15);
    assert_eq!(times.coordinate(5).unwrap().cycle, 1);
    assert_abs_diff_eq!(
        times.coordinate(5).unwrap().within_cycle,
        0.5,
        epsilon = 1.0e-15
    );
    assert_eq!(times.coordinate(8).unwrap().cycle, 2);
    assert!(matches!(
        times.coordinate(9).unwrap_err(),
        QmbedError::InvalidOptions(_)
    ));
}

#[test]
fn dynamical_correlator_matches_a_two_level_lehmann_phase() {
    let hamiltonian = diagonal(&[-1.0, 1.0]);
    let sigma_x = Operator::from_dense(
        2,
        2,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap();
    let times = vec![0.0, 0.3, 0.8];
    let values = dynamical_correlator(
        &hamiltonian,
        &[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
        &sigma_x,
        &sigma_x,
        EvolutionOptions {
            times: times.clone(),
            krylov_dimension: 8,
            tolerance: 1.0e-12,
            max_substeps: 100,
            hamiltonian: true,
        },
    )
    .unwrap();
    for (value, time) in values.iter().zip(times) {
        let expected = Complex64::new(0.0, -2.0 * time).exp();
        assert_abs_diff_eq!(value.re, expected.re, epsilon = 1.0e-12);
        assert_abs_diff_eq!(value.im, expected.im, epsilon = 1.0e-12);
    }
}
