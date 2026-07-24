use std::sync::Arc;

use approx::assert_abs_diff_eq;
use qmbed::Complex64;
use qmbed::block::{BlockOps, DynamicBlockOps, block_diag_hamiltonian};
use qmbed::operator::{
    Dynamic, DynamicComponent, Hamiltonian, LinearOperator, MatrixFormat, Operator,
    TimeDependentOperator,
};

fn diagonal(values: &[f64]) -> Operator {
    let dimension = values.len();
    let mut dense = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    for (index, value) in values.iter().enumerate() {
        dense[index * dimension + index] = Complex64::new(*value, 0.0);
    }
    Operator::from_dense(dimension, dimension, dense).unwrap()
}

#[test]
fn delayed_and_materialized_static_blocks_agree() {
    let blocks: Vec<Arc<dyn LinearOperator>> =
        vec![Arc::new(diagonal(&[-2.0, 1.0])), Arc::new(diagonal(&[3.0]))];
    let delayed = BlockOps::new(blocks.clone()).unwrap();
    let materialized = block_diag_hamiltonian(blocks, MatrixFormat::Csc).unwrap();
    assert_eq!(delayed.shape(), (3, 3));
    let input = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(-0.5, 0.0),
        Complex64::new(2.0, 0.0),
    ];
    let mut left = vec![Complex64::new(0.0, 0.0); 3];
    let mut right = left.clone();
    delayed.apply(&input, &mut left).unwrap();
    materialized.apply(&input, &mut right).unwrap();
    assert_eq!(left, right);
    assert_eq!(
        delayed.materialize(MatrixFormat::Csr).unwrap().to_dense(),
        materialized.to_dense()
    );
}

#[test]
fn dynamic_blocks_apply_each_sector_at_the_same_time() {
    let first = Hamiltonian::<Dynamic>::new(
        diagonal(&[0.0]),
        vec![DynamicComponent::new(diagonal(&[2.0]), |time| {
            Complex64::new(time, 0.0)
        })],
    )
    .unwrap();
    let second = Hamiltonian::<Dynamic>::new(
        diagonal(&[1.0]),
        vec![DynamicComponent::new(diagonal(&[-1.0]), |time| {
            Complex64::new(time, 0.0)
        })],
    )
    .unwrap();
    let blocks: Vec<Arc<dyn TimeDependentOperator>> = vec![Arc::new(first), Arc::new(second)];
    let dynamic = DynamicBlockOps::new(blocks).unwrap();
    let mut output = vec![Complex64::new(0.0, 0.0); 2];
    dynamic
        .apply_at(
            0.25,
            &[Complex64::new(2.0, 0.0), Complex64::new(4.0, 0.0)],
            &mut output,
        )
        .unwrap();
    assert_abs_diff_eq!(output[0].re, 1.0, epsilon = 1.0e-12);
    assert_abs_diff_eq!(output[1].re, 3.0, epsilon = 1.0e-12);
}
