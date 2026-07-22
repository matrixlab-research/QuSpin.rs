use std::sync::Arc;

use num_complex::Complex64;

use crate::operator::{LinearOperator, MatrixFormat, check_apply_shape, materialize_dense};
use crate::{QuSpinError, Result};

/// Matrix-free Lindblad generator over column-major vectorized density matrices.
pub struct LindbladGenerator {
    hamiltonian: Arc<dyn LinearOperator>,
    jumps: Vec<Arc<dyn LinearOperator>>,
    dimension: usize,
}

impl LindbladGenerator {
    pub fn new(
        hamiltonian: Arc<dyn LinearOperator>,
        jumps: Vec<Arc<dyn LinearOperator>>,
    ) -> Result<Self> {
        let shape = hamiltonian.shape();
        if shape.0 != shape.1 {
            return Err(QuSpinError::DimensionMismatch(
                "the Lindblad Hamiltonian must be square".into(),
            ));
        }
        if jumps.iter().any(|jump| jump.shape() != shape) {
            return Err(QuSpinError::DimensionMismatch(
                "all Lindblad jumps must match the Hamiltonian".into(),
            ));
        }
        Ok(Self {
            hamiltonian,
            jumps,
            dimension: shape.0,
        })
    }
}

fn multiply(
    left: &[Complex64],
    right: &[Complex64],
    dimension: usize,
) -> Vec<Complex64> {
    let mut product = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    for row in 0..dimension {
        for middle in 0..dimension {
            for column in 0..dimension {
                product[row * dimension + column] += left[row * dimension + middle]
                    * right[middle * dimension + column];
            }
        }
    }
    product
}

fn adjoint(matrix: &[Complex64], dimension: usize) -> Vec<Complex64> {
    let mut result = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    for row in 0..dimension {
        for column in 0..dimension {
            result[row * dimension + column] = matrix[column * dimension + row].conj();
        }
    }
    result
}

fn column_major_to_row_major(vector: &[Complex64], dimension: usize) -> Vec<Complex64> {
    let mut matrix = vec![Complex64::new(0.0, 0.0); vector.len()];
    for row in 0..dimension {
        for column in 0..dimension {
            matrix[row * dimension + column] = vector[row + column * dimension];
        }
    }
    matrix
}

fn row_major_to_column_major(matrix: &[Complex64], dimension: usize) -> Vec<Complex64> {
    let mut vector = vec![Complex64::new(0.0, 0.0); matrix.len()];
    for row in 0..dimension {
        for column in 0..dimension {
            vector[row + column * dimension] = matrix[row * dimension + column];
        }
    }
    vector
}

impl LinearOperator for LindbladGenerator {
    fn shape(&self) -> (usize, usize) {
        let size = self.dimension * self.dimension;
        (size, size)
    }

    fn format(&self) -> MatrixFormat {
        MatrixFormat::MatrixFree
    }

    fn apply(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        check_apply_shape(self.shape(), input, output)?;
        let dimension = self.dimension;
        let density = column_major_to_row_major(input, dimension);
        let hamiltonian = materialize_dense(self.hamiltonian.as_ref())?;
        let h_rho = multiply(&hamiltonian, &density, dimension);
        let rho_h = multiply(&density, &hamiltonian, dimension);
        let mut derivative: Vec<_> = h_rho
            .iter()
            .zip(&rho_h)
            .map(|(left, right)| Complex64::new(0.0, -1.0) * (*left - *right))
            .collect();
        for jump in &self.jumps {
            let jump = materialize_dense(jump.as_ref())?;
            let jump_adjoint = adjoint(&jump, dimension);
            let jump_product = multiply(&jump_adjoint, &jump, dimension);
            let gain = multiply(&multiply(&jump, &density, dimension), &jump_adjoint, dimension);
            let loss_left = multiply(&jump_product, &density, dimension);
            let loss_right = multiply(&density, &jump_product, dimension);
            for index in 0..derivative.len() {
                derivative[index] +=
                    gain[index] - 0.5 * (loss_left[index] + loss_right[index]);
            }
        }
        output.copy_from_slice(&row_major_to_column_major(&derivative, dimension));
        Ok(())
    }
}
