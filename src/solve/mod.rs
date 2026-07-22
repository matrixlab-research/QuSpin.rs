use std::cmp::Ordering;

use nalgebra::{DMatrix, SymmetricEigen};
use num_complex::Complex64;

use crate::operator::{LinearOperator, materialize_dense};
use crate::{QuSpinError, Result};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpectrumTarget {
    SmallestAlgebraic,
    LargestAlgebraic,
    LargestMagnitude,
    Shift(f64),
}

#[derive(Clone, Debug, PartialEq)]
pub struct EigshOptions {
    pub eigenpairs: usize,
    pub target: SpectrumTarget,
    pub krylov_dimension: Option<usize>,
    pub tolerance: f64,
    pub max_iterations: usize,
    pub seed: u64,
}

impl EigshOptions {
    pub fn smallest_algebraic(eigenpairs: usize) -> Self {
        Self {
            eigenpairs,
            target: SpectrumTarget::SmallestAlgebraic,
            krylov_dimension: None,
            tolerance: 1.0e-10,
            max_iterations: 1_000,
            seed: 0,
        }
    }

    fn validate(&self, dimension: usize) -> Result<()> {
        if self.eigenpairs == 0 || self.eigenpairs >= dimension {
            return Err(QuSpinError::InvalidOptions(
                "eigenpairs must be positive and smaller than the operator dimension".into(),
            ));
        }
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 || self.max_iterations == 0 {
            return Err(QuSpinError::InvalidOptions(
                "tolerance and max_iterations must be positive".into(),
            ));
        }
        if self
            .krylov_dimension
            .is_some_and(|size| size <= self.eigenpairs || size > dimension)
        {
            return Err(QuSpinError::InvalidOptions(
                "krylov_dimension must exceed eigenpairs and not exceed dimension".into(),
            ));
        }
        if matches!(self.target, SpectrumTarget::Shift(value) if !value.is_finite()) {
            return Err(QuSpinError::InvalidOptions("shift must be finite".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Eigensystem {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Vec<Vec<Complex64>>,
    pub residuals: Vec<f64>,
    pub iterations: usize,
}

pub(crate) fn real_symmetric_eigenpairs_all(
    operator: &(impl LinearOperator + ?Sized),
) -> Result<(Vec<f64>, Vec<Vec<Complex64>>)> {
    let shape = operator.shape();
    if shape.0 != shape.1 {
        return Err(QuSpinError::DimensionMismatch(
            "a square operator is required".into(),
        ));
    }
    let dense = materialize_dense(operator)?;
    if dense.iter().any(|value| value.im.abs() > 1.0e-13) {
        return Err(QuSpinError::UnsupportedBackend(
            "complex Hermitian eigensystems are not active in the first solver backend".into(),
        ));
    }
    let dimension = shape.0;
    for row in 0..dimension {
        for column in 0..dimension {
            if (dense[row * dimension + column].re - dense[column * dimension + row].re).abs()
                > 1.0e-12
            {
                return Err(QuSpinError::NonHermitian);
            }
        }
    }
    let matrix = DMatrix::from_fn(dimension, dimension, |row, column| {
        dense[row * dimension + column].re
    });
    let decomposition = SymmetricEigen::new(matrix);
    let values = decomposition.eigenvalues.as_slice().to_vec();
    let vectors = (0..dimension)
        .map(|column| {
            (0..dimension)
                .map(|row| Complex64::new(decomposition.eigenvectors[(row, column)], 0.0))
                .collect()
        })
        .collect();
    Ok((values, vectors))
}

fn residual_norm(
    operator: &(impl LinearOperator + ?Sized),
    eigenvalue: f64,
    vector: &[Complex64],
) -> Result<f64> {
    let mut applied = vec![Complex64::new(0.0, 0.0); vector.len()];
    operator.apply(vector, &mut applied)?;
    Ok(applied
        .iter()
        .zip(vector)
        .map(|(actual, component)| (*actual - eigenvalue * *component).norm_sqr())
        .sum::<f64>()
        .sqrt())
}

/// Selected Hermitian eigenpairs.
///
/// The first backend uses a dense real-symmetric decomposition for semantic
/// closure. The public interface is already operator-based so a Lanczos and
/// shift-invert backend can replace it without changing callers.
pub fn eigsh<O>(operator: &O, options: EigshOptions) -> Result<Eigensystem>
where
    O: LinearOperator + ?Sized,
{
    let shape = operator.shape();
    if shape.0 != shape.1 {
        return Err(QuSpinError::DimensionMismatch(
            "eigsh requires a square operator".into(),
        ));
    }
    options.validate(shape.0)?;
    let (values, vectors) = real_symmetric_eigenpairs_all(operator)?;
    let mut indices: Vec<_> = (0..values.len()).collect();
    indices.sort_by(|&left, &right| {
        let left_value = values[left];
        let right_value = values[right];
        let ordering = match options.target {
            SpectrumTarget::SmallestAlgebraic => left_value.total_cmp(&right_value),
            SpectrumTarget::LargestAlgebraic => right_value.total_cmp(&left_value),
            SpectrumTarget::LargestMagnitude => right_value.abs().total_cmp(&left_value.abs()),
            SpectrumTarget::Shift(shift) => (left_value - shift)
                .abs()
                .total_cmp(&(right_value - shift).abs()),
        };
        if ordering == Ordering::Equal {
            left.cmp(&right)
        } else {
            ordering
        }
    });
    indices.truncate(options.eigenpairs);
    let eigenvalues: Vec<_> = indices.iter().map(|&index| values[index]).collect();
    let eigenvectors: Vec<_> = indices
        .iter()
        .map(|&index| vectors[index].clone())
        .collect();
    let residuals = eigenvalues
        .iter()
        .zip(&eigenvectors)
        .map(|(&value, vector)| residual_norm(operator, value, vector))
        .collect::<Result<Vec<_>>>()?;
    Ok(Eigensystem {
        eigenvalues,
        eigenvectors,
        residuals,
        iterations: 1,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvolutionOptions {
    pub times: Vec<f64>,
    pub krylov_dimension: usize,
    pub tolerance: f64,
    pub max_substeps: usize,
    pub hamiltonian: bool,
}

impl EvolutionOptions {
    fn validate(&self) -> Result<()> {
        if self.times.is_empty()
            || self.times.iter().any(|time| !time.is_finite())
            || self.times.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err(QuSpinError::InvalidOptions(
                "times must be a nonempty finite nondecreasing grid".into(),
            ));
        }
        if self.krylov_dimension == 0
            || !self.tolerance.is_finite()
            || self.tolerance <= 0.0
            || self.max_substeps == 0
        {
            return Err(QuSpinError::InvalidOptions(
                "evolution controls must be positive".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct StateTrajectory {
    pub times: Vec<f64>,
    pub states: Vec<Vec<Complex64>>,
}

fn vector_norm(vector: &[Complex64]) -> f64 {
    vector.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt()
}

pub(crate) fn expm_action(
    operator: &(impl LinearOperator + ?Sized),
    initial: &[Complex64],
    interval: f64,
    options: &EvolutionOptions,
) -> Result<Vec<Complex64>> {
    let shape = operator.shape();
    if shape.0 != shape.1 || initial.len() != shape.0 {
        return Err(QuSpinError::DimensionMismatch(
            "evolution requires a square operator matching the state".into(),
        ));
    }
    if interval == 0.0 {
        return Ok(initial.to_vec());
    }
    let requested_steps = interval.abs().ceil().max(1.0) as usize;
    if requested_steps > options.max_substeps {
        return Err(QuSpinError::NonConvergence {
            iterations: options.max_substeps,
            residual: interval.abs(),
        });
    }
    let step = interval / requested_steps as f64;
    let factor = if options.hamiltonian {
        Complex64::new(0.0, -step)
    } else {
        Complex64::new(step, 0.0)
    };
    let mut state = initial.to_vec();
    let mut applied = vec![Complex64::new(0.0, 0.0); shape.0];
    for _ in 0..requested_steps {
        let mut sum = state.clone();
        let mut term = state.clone();
        for order in 1..=options.krylov_dimension {
            operator.apply(&term, &mut applied)?;
            let scale = factor / order as f64;
            for (next, value) in term.iter_mut().zip(&applied) {
                *next = scale * *value;
            }
            for (total, value) in sum.iter_mut().zip(&term) {
                *total += *value;
            }
            if vector_norm(&term) <= options.tolerance * vector_norm(&sum).max(1.0) {
                break;
            }
            if order == options.krylov_dimension {
                return Err(QuSpinError::NonConvergence {
                    iterations: order,
                    residual: vector_norm(&term),
                });
            }
        }
        state = sum;
    }
    Ok(state)
}

/// Time evolution on an arbitrary square stored or matrix-free operator.
pub fn evolve<O>(
    operator: &O,
    initial: &[Complex64],
    options: EvolutionOptions,
) -> Result<StateTrajectory>
where
    O: LinearOperator + ?Sized,
{
    options.validate()?;
    let shape = operator.shape();
    if shape.0 != shape.1 || initial.len() != shape.0 {
        return Err(QuSpinError::DimensionMismatch(
            "evolution operator and initial state do not match".into(),
        ));
    }
    let mut states = Vec::with_capacity(options.times.len());
    let mut state = initial.to_vec();
    let mut previous_time = 0.0;
    for &time in &options.times {
        state = expm_action(operator, &state, time - previous_time, &options)?;
        states.push(state.clone());
        previous_time = time;
    }
    Ok(StateTrajectory {
        times: options.times,
        states,
    })
}
