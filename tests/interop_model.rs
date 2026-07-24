use approx::assert_abs_diff_eq;
use qmbed::basis::{Basis, BosonBasis1D, PackedBasis, SpinBasis1D};
use qmbed::interop::PackedEdModel;
use qmbed::operator::{
    Coupling, LinearOperator, LocalOperator, MatrixFormat, OpProduct, OperatorBuilder, OperatorSpec,
};
use qmbed::solve::EighOptions;
use std::sync::Arc;

#[test]
fn packed_basis_preserves_concrete_operator_semantics() {
    let basis = SpinBasis1D::builder(3).up(1).build().unwrap();
    let packed = PackedBasis::from(basis.clone());
    let term = OperatorSpec::from_product(
        OpProduct::new([LocalOperator::Z]).unwrap(),
        (0..3).map(|site| Coupling::new(site as f64 + 1.0, vec![site])),
    )
    .unwrap();

    let concrete = OperatorBuilder::on(&basis)
        .term(term.clone())
        .build(MatrixFormat::Csc)
        .unwrap();
    let erased = OperatorBuilder::on(&packed)
        .term(term)
        .build(MatrixFormat::Csc)
        .unwrap();

    assert_eq!(packed.len(), basis.len());
    assert_eq!(erased.triplets(), concrete.triplets());
}

#[test]
fn reversed_packed_basis_reorders_states_and_operator_rows_together() {
    let natural = PackedBasis::from(SpinBasis1D::builder(1).build().unwrap());
    let reversed = natural.clone().reversed();
    let term = OperatorSpec::from_product(
        OpProduct::new([LocalOperator::Y]).unwrap(),
        [Coupling::new(1.0, vec![0])],
    )
    .unwrap();
    let natural_operator = OperatorBuilder::on(&natural)
        .term(term.clone())
        .build(MatrixFormat::Dense)
        .unwrap();
    let reversed_operator = OperatorBuilder::on(&reversed)
        .term(term)
        .build(MatrixFormat::Dense)
        .unwrap();

    assert_eq!(natural.state(0).unwrap(), reversed.state(1).unwrap());
    assert_eq!(natural.state(1).unwrap(), reversed.state(0).unwrap());
    let natural_dense = natural_operator.to_dense();
    let reversed_dense = reversed_operator.to_dense();
    assert_eq!(reversed_dense[0], natural_dense[3]);
    assert_eq!(reversed_dense[1], natural_dense[2]);
    assert_eq!(reversed_dense[2], natural_dense[1]);
    assert_eq!(reversed_dense[3], natural_dense[0]);
}

#[test]
fn operator_site_permutation_changes_labels_not_assembly_code() {
    let basis = SpinBasis1D::builder(2).build().unwrap();
    let left = OperatorSpec::from_product(
        OpProduct::new([LocalOperator::X, LocalOperator::Identity]).unwrap(),
        [Coupling::new(1.0, vec![0, 1])],
    )
    .unwrap();
    let right = left.with_site_permutation(&[1, 0]).unwrap();
    let left = OperatorBuilder::on(&basis)
        .term(left)
        .build(MatrixFormat::Dense)
        .unwrap();
    let right = OperatorBuilder::on(&basis)
        .term(right)
        .build(MatrixFormat::Dense)
        .unwrap();

    assert_eq!(left.to_dense()[4].re, 0.5);
    assert_eq!(right.to_dense()[8].re, 0.5);
    assert_ne!(left.triplets(), right.triplets());
}

#[test]
fn packed_model_reuses_one_spec_for_states_matrix_and_eigh() {
    let basis = BosonBasis1D::builder(1, 4).build().unwrap();
    let terms = ["+", "-", "n"].into_iter().map(|operator| {
        let local = match operator {
            "+" => LocalOperator::Raising,
            "-" => LocalOperator::Lowering,
            "n" => LocalOperator::Number,
            _ => unreachable!(),
        };
        let coefficient = if operator == "n" { 0.25 } else { 1.0 };
        OperatorSpec::from_product(
            OpProduct::new([local]).unwrap(),
            [Coupling::new(coefficient, vec![0])],
        )
        .unwrap()
    });
    let model = PackedEdModel::new(basis, terms);

    assert_eq!(model.dimension(), 4);
    assert_eq!(model.states().unwrap(), vec![0, 1, 2, 3]);
    let operator = model.materialize(MatrixFormat::Csc).unwrap();
    let result = model
        .eigh(EighOptions {
            return_eigenvectors: false,
        })
        .unwrap();

    assert_eq!(operator.shape(), (4, 4));
    assert_eq!(result.eigenvalues.len(), 4);
    assert!(result.eigenvectors.is_empty());
    assert_abs_diff_eq!(result.eigenvalues[0], -1.885007105857148, epsilon = 1.0e-12);
}

#[test]
fn packed_model_caches_each_materialized_format() {
    let basis = SpinBasis1D::builder(2).build().unwrap();
    let term = OperatorSpec::from_product(
        OpProduct::new([LocalOperator::Z]).unwrap(),
        [Coupling::new(1.0, vec![0])],
    )
    .unwrap();
    let model = PackedEdModel::new(basis, [term]);

    let first = model.materialized(MatrixFormat::Csc).unwrap();
    let second = model.materialized(MatrixFormat::Csc).unwrap();
    let dense = model.materialized(MatrixFormat::Dense).unwrap();

    assert!(Arc::ptr_eq(&first, &second));
    assert!(!Arc::ptr_eq(&first, &dense));
    assert_eq!(first.to_dense(), dense.to_dense());
}

#[test]
fn transformed_model_does_not_reuse_a_stale_operator() {
    let basis = SpinBasis1D::builder(2).build().unwrap();
    let term = OperatorSpec::from_product(
        OpProduct::new([LocalOperator::Z]).unwrap(),
        [Coupling::new(1.0, vec![0])],
    )
    .unwrap();
    let model = PackedEdModel::new(basis, [term]);
    let original = model.materialized(MatrixFormat::Csc).unwrap();
    let permuted_model = model.with_site_permutation(&[1, 0]).unwrap();
    let permuted = permuted_model.materialized(MatrixFormat::Csc).unwrap();

    assert!(!Arc::ptr_eq(&original, &permuted));
    assert_ne!(original.triplets(), permuted.triplets());
}
