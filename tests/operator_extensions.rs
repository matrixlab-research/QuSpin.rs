use std::collections::HashMap;

use approx::assert_abs_diff_eq;
use quspin::basis::SpinBasis1D;
use quspin::operator::{
    Coupling, DynamicTerm, LinearOperator, MatrixFormat, Operator, OperatorBuilder, OperatorTerm,
    QuantumComponent, QuantumOperator, TimeDependentOperator, anticommutator, commutator,
};
use quspin::{Complex64, QuSpinError};

fn assert_complex_close(actual: Complex64, expected: Complex64) {
    assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-12);
    assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-12);
}

#[test]
fn dynamic_hamiltonian_evaluation_and_action_share_one_semantics() {
    let basis = SpinBasis1D::builder(1).pauli(true).build().unwrap();
    let static_term = OperatorTerm::new("z", [Coupling::new(0.25, vec![0])]).unwrap();
    let driven_term = DynamicTerm::new(
        OperatorTerm::new("x", [Coupling::new(1.0, vec![0])]).unwrap(),
        |time| Complex64::new(time.sin(), 0.0),
    );
    let hamiltonian = OperatorBuilder::on(&basis)
        .term(static_term)
        .build_dynamic([driven_term], MatrixFormat::Csc)
        .unwrap();
    assert_eq!(hamiltonian.dynamic_components(), 1);

    let time = std::f64::consts::FRAC_PI_2;
    let evaluated = hamiltonian.evaluate(time, MatrixFormat::Dense).unwrap();
    let mut matrix_free = vec![Complex64::new(0.0, 0.0); 2];
    hamiltonian
        .apply_at(
            time,
            &[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            &mut matrix_free,
        )
        .unwrap();
    let mut materialized = vec![Complex64::new(0.0, 0.0); 2];
    evaluated
        .apply(
            &[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            &mut materialized,
        )
        .unwrap();
    for (actual, expected) in matrix_free.into_iter().zip(materialized) {
        assert_complex_close(actual, expected);
    }
}

#[test]
fn quantum_operator_uses_required_and_default_parameters() {
    let identity = Operator::from_dense(
        2,
        2,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    )
    .unwrap();
    let sigma_z = Operator::from_dense(
        2,
        2,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
        ],
    )
    .unwrap();
    let parameterized = QuantumOperator::new([
        QuantumComponent::required("field", sigma_z),
        QuantumComponent::with_default("offset", identity, 2.0),
    ])
    .unwrap();

    let parameters = HashMap::from([("field".to_string(), Complex64::new(3.0, 0.0))]);
    let evaluated = parameterized
        .evaluate(&parameters, MatrixFormat::Csr)
        .unwrap();
    let diagonal = evaluated.diagonal();
    assert_complex_close(diagonal[0], Complex64::new(5.0, 0.0));
    assert_complex_close(diagonal[1], Complex64::new(-1.0, 0.0));

    let missing = parameterized
        .evaluate(&HashMap::new(), MatrixFormat::Dense)
        .unwrap_err();
    assert!(matches!(missing, QuSpinError::InvalidOptions(_)));
}

#[test]
fn operator_algebra_obeys_pauli_identities_and_format_conversion() {
    let x = Operator::from_dense(
        2,
        2,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap()
    .converted(MatrixFormat::Csc)
    .unwrap();
    let y = Operator::from_dense(
        2,
        2,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap()
    .converted(MatrixFormat::Csc)
    .unwrap();

    let xy_commutator = commutator(&x, &y).unwrap().to_dense();
    assert_complex_close(xy_commutator[0], Complex64::new(0.0, 2.0));
    assert_complex_close(xy_commutator[3], Complex64::new(0.0, -2.0));
    let xy_anticommutator = anticommutator(&x, &y).unwrap();
    assert_eq!(xy_anticommutator.nnz(), 0);
    let identity = x.pow(2).unwrap().to_dense();
    assert_complex_close(identity[0], Complex64::new(1.0, 0.0));
    assert_complex_close(identity[3], Complex64::new(1.0, 0.0));
    assert_eq!(x.adjoint().unwrap().to_dense(), x.to_dense());
}
