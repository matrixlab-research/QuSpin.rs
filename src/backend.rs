//! Coarse-grained numerical backend boundary.
//!
//! Physics-facing modules own basis, operator, and workflow semantics. This
//! module owns conversions into third-party dense and sparse kernels so solver
//! algorithms do not depend directly on a concrete linear-algebra crate.

use faer::Mat;
use faer::linalg::solvers::Solve;
use faer::sparse::{SparseColMat, Triplet};
use nalgebra::{DMatrix, SymmetricEigen, linalg::Schur};
use num_complex::Complex64;

use crate::{QuSpinError, Result};

/// Backend-neutral result of a complete Hermitian eigendecomposition.
pub(crate) struct HermitianEigensystem {
    pub(crate) eigenvalues: Vec<f64>,
    /// Eigenvectors are stored one column per outer vector.
    pub(crate) eigenvectors: Vec<Vec<Complex64>>,
}

/// Backend-neutral result of a complete complex eigendecomposition.
pub(crate) struct ComplexEigensystem {
    pub(crate) eigenvalues: Vec<Complex64>,
    /// Right eigenvectors are stored one column per outer vector.
    pub(crate) eigenvectors: Vec<Vec<Complex64>>,
}

fn validate_square_dense(values: &[Complex64], dimension: usize) -> Result<()> {
    if values.len() != dimension.saturating_mul(dimension) {
        return Err(QuSpinError::DimensionMismatch(format!(
            "dense backend expected {} values for a {dimension}x{dimension} matrix, got {}",
            dimension.saturating_mul(dimension),
            values.len()
        )));
    }
    Ok(())
}

/// Complete eigendecomposition of a row-major Hermitian matrix.
pub(crate) fn hermitian_eigenpairs(
    values: &[Complex64],
    dimension: usize,
) -> Result<HermitianEigensystem> {
    validate_square_dense(values, dimension)?;
    let is_real = values.iter().all(|value| value.im.abs() <= 1.0e-14);
    if is_real {
        let matrix = DMatrix::<f64>::from_fn(dimension, dimension, |row, column| {
            values[row * dimension + column].re
        });
        let decomposition = SymmetricEigen::new(matrix);
        let mut eigenpairs: Vec<_> = (0..dimension)
            .map(|column| {
                (
                    decomposition.eigenvalues[column],
                    (0..dimension)
                        .map(|row| Complex64::new(decomposition.eigenvectors[(row, column)], 0.0))
                        .collect(),
                )
            })
            .collect();
        eigenpairs.sort_by(|left, right| left.0.total_cmp(&right.0));
        let (eigenvalues, eigenvectors) = eigenpairs.into_iter().unzip();
        return Ok(HermitianEigensystem {
            eigenvalues,
            eigenvectors,
        });
    }

    let matrix = DMatrix::<Complex64>::from_fn(dimension, dimension, |row, column| {
        values[row * dimension + column]
    });
    let (vectors, triangular) = Schur::new(matrix).unpack();
    let mut eigenpairs = Vec::with_capacity(dimension);
    for column in 0..dimension {
        let value = triangular[(column, column)];
        if value.im.abs() > 1.0e-10 {
            return Err(QuSpinError::NonHermitian);
        }
        let vector = (0..dimension).map(|row| vectors[(row, column)]).collect();
        eigenpairs.push((value.re, vector));
    }
    eigenpairs.sort_by(|left, right| left.0.total_cmp(&right.0));
    let (eigenvalues, eigenvectors) = eigenpairs.into_iter().unzip();
    Ok(HermitianEigensystem {
        eigenvalues,
        eigenvectors,
    })
}

/// Complete right eigendecomposition of a row-major complex matrix.
pub(crate) fn complex_eigenpairs(
    values: &[Complex64],
    dimension: usize,
) -> Result<ComplexEigensystem> {
    validate_square_dense(values, dimension)?;
    let matrix = Mat::<Complex64>::from_fn(dimension, dimension, |row, column| {
        values[row * dimension + column]
    });
    let decomposition = matrix.eigen().map_err(|error| {
        QuSpinError::UnsupportedBackend(format!("complex eigendecomposition failed: {error:?}"))
    })?;
    Ok(ComplexEigensystem {
        eigenvalues: (0..dimension)
            .map(|index| decomposition.S()[index])
            .collect(),
        eigenvectors: (0..dimension)
            .map(|column| {
                (0..dimension)
                    .map(|row| decomposition.U()[(row, column)])
                    .collect()
            })
            .collect(),
    })
}

/// Form `exp(coefficient * H)` for a row-major Hermitian matrix.
pub(crate) fn hermitian_exponential(
    values: &[Complex64],
    dimension: usize,
    coefficient: Complex64,
) -> Result<Vec<Complex64>> {
    validate_square_dense(values, dimension)?;
    if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
        return Err(QuSpinError::InvalidOptions(
            "matrix exponential coefficient must be finite".into(),
        ));
    }
    let eigensystem = hermitian_eigenpairs(values, dimension)?;
    let vectors = Mat::<Complex64>::from_fn(dimension, dimension, |row, column| {
        eigensystem.eigenvectors[column][row]
    });
    let weighted = Mat::<Complex64>::from_fn(dimension, dimension, |row, column| {
        vectors[(row, column)] * (coefficient * eigensystem.eigenvalues[column]).exp()
    });
    let exponential = &weighted * vectors.adjoint();
    let mut output = Vec::with_capacity(dimension.saturating_mul(dimension));
    for row in 0..dimension {
        for column in 0..dimension {
            output.push(exponential[(row, column)]);
        }
    }
    Ok(output)
}

/// Multiply two row-major square complex matrices.
pub(crate) fn square_matmul(
    left: &[Complex64],
    right: &[Complex64],
    dimension: usize,
) -> Result<Vec<Complex64>> {
    validate_square_dense(left, dimension)?;
    validate_square_dense(right, dimension)?;
    let left = Mat::<Complex64>::from_fn(dimension, dimension, |row, column| {
        left[row * dimension + column]
    });
    let right = Mat::<Complex64>::from_fn(dimension, dimension, |row, column| {
        right[row * dimension + column]
    });
    let product = &left * &right;
    let mut output = Vec::with_capacity(dimension.saturating_mul(dimension));
    for row in 0..dimension {
        for column in 0..dimension {
            output.push(product[(row, column)]);
        }
    }
    Ok(output)
}

/// Reusable factorization of `(A - shift I)`.
pub trait ShiftedLinearSolver: Send + Sync {
    fn solve(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()>;

    fn supports_real(&self) -> bool {
        false
    }

    fn solve_real(&self, _input: &[f64], _output: &mut [f64]) -> Result<()> {
        Err(QuSpinError::UnsupportedBackend(
            "shifted factorization does not support real right-hand sides".into(),
        ))
    }
}

enum FaerShiftedFactorization {
    Real(faer::sparse::linalg::solvers::Lu<usize, f64>),
    Complex(faer::sparse::linalg::solvers::Lu<usize, Complex64>),
}

struct FaerShiftedSolver {
    factorization: FaerShiftedFactorization,
    dimension: usize,
}

impl ShiftedLinearSolver for FaerShiftedSolver {
    fn solve(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        if input.len() != self.dimension || output.len() != self.dimension {
            return Err(QuSpinError::DimensionMismatch(
                "shifted solve input or output length does not match".into(),
            ));
        }
        match &self.factorization {
            FaerShiftedFactorization::Real(factorization) => {
                let mut real = faer::Col::from_fn(self.dimension, |index| input[index].re);
                let mut imaginary = faer::Col::from_fn(self.dimension, |index| input[index].im);
                factorization.solve_in_place(real.as_mut());
                factorization.solve_in_place(imaginary.as_mut());
                for (index, value) in output.iter_mut().enumerate() {
                    *value = Complex64::new(real[index], imaginary[index]);
                }
            }
            FaerShiftedFactorization::Complex(factorization) => {
                let mut right_hand_side = faer::Col::from_fn(self.dimension, |index| input[index]);
                factorization.solve_in_place(right_hand_side.as_mut());
                for (index, value) in output.iter_mut().enumerate() {
                    *value = right_hand_side[index];
                }
            }
        }
        Ok(())
    }

    fn supports_real(&self) -> bool {
        matches!(self.factorization, FaerShiftedFactorization::Real(_))
    }

    fn solve_real(&self, input: &[f64], output: &mut [f64]) -> Result<()> {
        if input.len() != self.dimension || output.len() != self.dimension {
            return Err(QuSpinError::DimensionMismatch(
                "real shifted solve input or output length does not match".into(),
            ));
        }
        let FaerShiftedFactorization::Real(factorization) = &self.factorization else {
            return Err(QuSpinError::UnsupportedBackend(
                "complex shifted factorization cannot use the real fast path".into(),
            ));
        };
        let mut right_hand_side = faer::Col::from_fn(self.dimension, |index| input[index]);
        factorization.solve_in_place(right_hand_side.as_mut());
        for (index, value) in output.iter_mut().enumerate() {
            *value = right_hand_side[index];
        }
        Ok(())
    }
}

/// Factor a canonical CSC matrix after applying a real diagonal shift.
pub(crate) fn factor_shifted_csc(
    dimension: usize,
    column_offsets: &[usize],
    row_indices: &[usize],
    values: &[Complex64],
    shift: f64,
    is_real: bool,
) -> Result<Box<dyn ShiftedLinearSolver>> {
    if column_offsets.len() != dimension + 1
        || row_indices.len() != values.len()
        || !shift.is_finite()
    {
        return Err(QuSpinError::DimensionMismatch(
            "invalid CSC storage for shifted factorization".into(),
        ));
    }

    if is_real {
        let mut triplets = Vec::with_capacity(values.len() + dimension);
        append_shifted_triplets(
            dimension,
            column_offsets,
            row_indices,
            values,
            shift,
            |row, column, value| {
                triplets.push(Triplet::new(row, column, value.re));
            },
        );
        let matrix =
            SparseColMat::<usize, f64>::try_new_from_triplets(dimension, dimension, &triplets)
                .map_err(|error| {
                    QuSpinError::UnsupportedBackend(format!(
                        "could not construct real sparse shifted matrix: {error}"
                    ))
                })?;
        let factorization = matrix.sp_lu().map_err(|_| QuSpinError::NonConvergence {
            iterations: 0,
            residual: f64::INFINITY,
        })?;
        return Ok(Box::new(FaerShiftedSolver {
            factorization: FaerShiftedFactorization::Real(factorization),
            dimension,
        }));
    }

    let mut triplets = Vec::with_capacity(values.len() + dimension);
    append_shifted_triplets(
        dimension,
        column_offsets,
        row_indices,
        values,
        shift,
        |row, column, value| {
            triplets.push(Triplet::new(row, column, value));
        },
    );
    let matrix =
        SparseColMat::<usize, Complex64>::try_new_from_triplets(dimension, dimension, &triplets)
            .map_err(|error| {
                QuSpinError::UnsupportedBackend(format!(
                    "could not construct complex sparse shifted matrix: {error}"
                ))
            })?;
    let factorization = matrix.sp_lu().map_err(|_| QuSpinError::NonConvergence {
        iterations: 0,
        residual: f64::INFINITY,
    })?;
    Ok(Box::new(FaerShiftedSolver {
        factorization: FaerShiftedFactorization::Complex(factorization),
        dimension,
    }))
}

fn append_shifted_triplets(
    dimension: usize,
    column_offsets: &[usize],
    row_indices: &[usize],
    values: &[Complex64],
    shift: f64,
    mut append: impl FnMut(usize, usize, Complex64),
) {
    for column in 0..dimension {
        let mut has_diagonal = false;
        for position in column_offsets[column]..column_offsets[column + 1] {
            let row = row_indices[position];
            let mut value = values[position];
            if row == column {
                value -= shift;
                has_diagonal = true;
            }
            append(row, column, value);
        }
        if !has_diagonal {
            append(column, column, Complex64::new(-shift, 0.0));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_backends_preserve_hermitian_and_general_eigenpairs() {
        let hermitian = [
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
            Complex64::new(0.0, -1.0),
            Complex64::new(1.0, 0.0),
        ];
        let result = hermitian_eigenpairs(&hermitian, 2).unwrap();
        assert!((result.eigenvalues[0] - 0.0).abs() < 1.0e-12);
        assert!((result.eigenvalues[1] - 2.0).abs() < 1.0e-12);

        let rotation = [
            Complex64::new(0.0, 0.0),
            Complex64::new(-1.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ];
        let result = complex_eigenpairs(&rotation, 2).unwrap();
        assert!(
            result
                .eigenvalues
                .iter()
                .all(|value| (value.norm() - 1.0).abs() < 1.0e-12)
        );
    }

    #[test]
    fn real_shifted_factorization_accepts_real_and_complex_rhs() {
        let solver = factor_shifted_csc(
            2,
            &[0, 2, 4],
            &[0, 1, 0, 1],
            &[
                Complex64::new(2.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(3.0, 0.0),
            ],
            0.0,
            true,
        )
        .unwrap();
        assert!(solver.supports_real());
        let mut real = [0.0; 2];
        solver.solve_real(&[1.0, 0.0], &mut real).unwrap();
        assert!((real[0] - 0.6).abs() < 1.0e-12);
        assert!((real[1] + 0.2).abs() < 1.0e-12);

        let mut complex = [Complex64::new(0.0, 0.0); 2];
        solver
            .solve(
                &[Complex64::new(1.0, 2.0), Complex64::new(0.0, 0.0)],
                &mut complex,
            )
            .unwrap();
        assert!((complex[0] - Complex64::new(0.6, 1.2)).norm() < 1.0e-12);
        assert!((complex[1] - Complex64::new(-0.2, -0.4)).norm() < 1.0e-12);
    }
}
