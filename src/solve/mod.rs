use std::cmp::Ordering;

use nalgebra::{DMatrix, DVector, SymmetricEigen};
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

fn inner(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    left.iter()
        .zip(right)
        .map(|(left_value, right_value)| left_value.conj() * *right_value)
        .sum()
}

fn vector_norm(vector: &[Complex64]) -> f64 {
    vector.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt()
}

fn normalize(vector: &mut [Complex64]) -> Result<()> {
    let norm = vector_norm(vector);
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(QuSpinError::NonConvergence {
            iterations: 0,
            residual: norm,
        });
    }
    for value in vector {
        *value /= norm;
    }
    Ok(())
}

fn deterministic_start(dimension: usize, seed: u64) -> Result<Vec<Complex64>> {
    let mut state = seed | 1;
    let mut vector = Vec::with_capacity(dimension);
    for _ in 0..dimension {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let mantissa = state >> 11;
        let value = mantissa as f64 / ((1_u64 << 53) as f64) - 0.5;
        vector.push(Complex64::new(value, 0.0));
    }
    normalize(&mut vector)?;
    Ok(vector)
}

fn shifted_apply<O>(
    operator: &O,
    shift: f64,
    input: &[Complex64],
    output: &mut [Complex64],
) -> Result<()>
where
    O: LinearOperator + ?Sized,
{
    operator.apply(input, output)?;
    for (value, input_value) in output.iter_mut().zip(input) {
        *value -= shift * *input_value;
    }
    Ok(())
}

fn gmres_shift_invert<O>(
    operator: &O,
    shift: f64,
    right_hand_side: &[Complex64],
    tolerance: f64,
    max_iterations: usize,
) -> Result<Vec<Complex64>>
where
    O: LinearOperator + ?Sized,
{
    let dimension = right_hand_side.len();
    let right_norm = vector_norm(right_hand_side);
    if right_norm <= f64::EPSILON {
        return Ok(vec![Complex64::new(0.0, 0.0); dimension]);
    }
    let restart = dimension.clamp(1, 256);
    let mut solution = vec![Complex64::new(0.0, 0.0); dimension];
    let mut residual = right_hand_side.to_vec();
    let mut applied = vec![Complex64::new(0.0, 0.0); dimension];
    let mut iterations = 0;

    while iterations < max_iterations {
        let beta = vector_norm(&residual);
        if beta <= tolerance * right_norm {
            return Ok(solution);
        }
        let mut first = residual.clone();
        for value in &mut first {
            *value /= beta;
        }
        let mut basis = vec![first];
        let cycle = restart.min(max_iterations - iterations);
        let mut hessenberg = vec![vec![Complex64::new(0.0, 0.0); cycle]; cycle + 1];
        let mut columns = 0;

        for column in 0..cycle {
            shifted_apply(operator, shift, &basis[column], &mut applied)?;
            for _ in 0..2 {
                for (row, vector) in basis.iter().enumerate() {
                    let overlap = inner(vector, &applied);
                    hessenberg[row][column] += overlap;
                    for (value, basis_value) in applied.iter_mut().zip(vector) {
                        *value -= overlap * *basis_value;
                    }
                }
            }
            let next_norm = vector_norm(&applied);
            hessenberg[column + 1][column] = Complex64::new(next_norm, 0.0);
            columns = column + 1;
            if next_norm <= 1.0e-14 {
                break;
            }
            let mut next = applied.clone();
            for value in &mut next {
                *value /= next_norm;
            }
            basis.push(next);
        }

        let mut normal = DMatrix::<Complex64>::zeros(columns, columns);
        let mut projected_rhs = DVector::<Complex64>::zeros(columns);
        for row in 0..columns {
            projected_rhs[row] = hessenberg[0][row].conj() * beta;
            for column in 0..columns {
                normal[(row, column)] = (0..=columns)
                    .map(|index| hessenberg[index][row].conj() * hessenberg[index][column])
                    .sum();
            }
        }
        let coefficients =
            normal
                .lu()
                .solve(&projected_rhs)
                .ok_or(QuSpinError::NonConvergence {
                    iterations,
                    residual: beta,
                })?;
        for (coefficient, vector) in coefficients.iter().zip(&basis) {
            for (value, basis_value) in solution.iter_mut().zip(vector) {
                *value += *coefficient * *basis_value;
            }
        }
        shifted_apply(operator, shift, &solution, &mut applied)?;
        for ((value, right_value), applied_value) in
            residual.iter_mut().zip(right_hand_side).zip(&applied)
        {
            *value = *right_value - *applied_value;
        }
        iterations += columns;
    }
    Err(QuSpinError::NonConvergence {
        iterations,
        residual: vector_norm(&residual),
    })
}

fn transformed_apply<O>(
    operator: &O,
    options: &EigshOptions,
    input: &[Complex64],
    output: &mut [Complex64],
) -> Result<()>
where
    O: LinearOperator + ?Sized,
{
    match options.target {
        SpectrumTarget::Shift(shift) => {
            let solved = gmres_shift_invert(
                operator,
                shift,
                input,
                (options.tolerance * 0.1).min(1.0e-10),
                options.max_iterations.max(128),
            )?;
            output.copy_from_slice(&solved);
            Ok(())
        }
        _ => operator.apply(input, output),
    }
}

fn select_indices(values: &[f64], target: SpectrumTarget, count: usize) -> Vec<usize> {
    let mut indices: Vec<_> = (0..values.len()).collect();
    indices.sort_by(|&left, &right| {
        let left_value = values[left];
        let right_value = values[right];
        let ordering = match target {
            SpectrumTarget::SmallestAlgebraic => left_value.total_cmp(&right_value),
            SpectrumTarget::LargestAlgebraic | SpectrumTarget::Shift(_) => {
                right_value.total_cmp(&left_value)
            }
            SpectrumTarget::LargestMagnitude => right_value.abs().total_cmp(&left_value.abs()),
        };
        ordering.then_with(|| left.cmp(&right))
    });
    indices.truncate(count);
    indices
}

fn lanczos_eigsh<O>(operator: &O, options: &EigshOptions) -> Result<Eigensystem>
where
    O: LinearOperator + ?Sized,
{
    let dimension = operator.shape().0;
    let requested_dimension = options
        .krylov_dimension
        .unwrap_or_else(|| (4 * options.eigenpairs + 24).max(48));
    let krylov_dimension = requested_dimension
        .min(options.max_iterations)
        .min(dimension);
    if krylov_dimension <= options.eigenpairs {
        return Err(QuSpinError::InvalidOptions(
            "the effective Krylov dimension must exceed eigenpairs".into(),
        ));
    }

    let mut basis = Vec::with_capacity(krylov_dimension);
    basis.push(deterministic_start(dimension, options.seed)?);
    let mut alphas = Vec::with_capacity(krylov_dimension);
    let mut betas = Vec::with_capacity(krylov_dimension.saturating_sub(1));
    let mut output = vec![Complex64::new(0.0, 0.0); dimension];

    for iteration in 0..krylov_dimension {
        transformed_apply(operator, options, &basis[iteration], &mut output)?;
        let alpha = inner(&basis[iteration], &output).re;
        alphas.push(alpha);
        for (value, basis_value) in output.iter_mut().zip(&basis[iteration]) {
            *value -= alpha * *basis_value;
        }
        if iteration > 0 {
            let beta = betas[iteration - 1];
            for (value, previous) in output.iter_mut().zip(&basis[iteration - 1]) {
                *value -= beta * *previous;
            }
        }

        // Full modified Gram-Schmidt reorthogonalization keeps multiple and
        // interior Ritz vectors reliable without the quadratic cost of a
        // second unconditional pass.
        for vector in &basis {
            let overlap = inner(vector, &output);
            for (value, basis_value) in output.iter_mut().zip(vector) {
                *value -= overlap * *basis_value;
            }
        }
        let beta = vector_norm(&output);
        if iteration + 1 == krylov_dimension || beta <= 1.0e-14 {
            break;
        }
        betas.push(beta);
        for value in &mut output {
            *value /= beta;
        }
        basis.push(output.clone());
    }

    if basis.len() <= options.eigenpairs {
        return Err(QuSpinError::NonConvergence {
            iterations: basis.len(),
            residual: f64::INFINITY,
        });
    }
    let size = basis.len();
    let mut tridiagonal = DMatrix::<f64>::zeros(size, size);
    for index in 0..size {
        tridiagonal[(index, index)] = alphas[index];
        if index + 1 < size {
            tridiagonal[(index, index + 1)] = betas[index];
            tridiagonal[(index + 1, index)] = betas[index];
        }
    }
    let decomposition = SymmetricEigen::new(tridiagonal);
    let transformed_values = decomposition.eigenvalues.as_slice();
    let transformed_target = if matches!(options.target, SpectrumTarget::Shift(_)) {
        SpectrumTarget::LargestMagnitude
    } else {
        options.target
    };
    let indices = select_indices(transformed_values, transformed_target, options.eigenpairs);

    let mut candidates = Vec::with_capacity(options.eigenpairs);
    for index in indices {
        let mut vector = vec![Complex64::new(0.0, 0.0); dimension];
        for (basis_index, basis_vector) in basis.iter().enumerate() {
            let coefficient = decomposition.eigenvectors[(basis_index, index)];
            for (value, basis_value) in vector.iter_mut().zip(basis_vector) {
                *value += coefficient * *basis_value;
            }
        }
        normalize(&mut vector)?;
        operator.apply(&vector, &mut output)?;
        let eigenvalue = inner(&vector, &output).re;
        let residual = output
            .iter()
            .zip(&vector)
            .map(|(actual, component)| (*actual - eigenvalue * *component).norm_sqr())
            .sum::<f64>()
            .sqrt();
        candidates.push((eigenvalue, vector, residual));
    }
    candidates.sort_by(|left, right| match options.target {
        SpectrumTarget::SmallestAlgebraic => left.0.total_cmp(&right.0),
        SpectrumTarget::LargestAlgebraic => right.0.total_cmp(&left.0),
        SpectrumTarget::LargestMagnitude => right.0.abs().total_cmp(&left.0.abs()),
        SpectrumTarget::Shift(shift) => (left.0 - shift).abs().total_cmp(&(right.0 - shift).abs()),
    });
    let residuals: Vec<_> = candidates.iter().map(|candidate| candidate.2).collect();
    let failure_residual = residuals.iter().copied().fold(0.0_f64, f64::max);
    let accepted_residual = options.tolerance.max(1.0e-7);
    if failure_residual > accepted_residual {
        return Err(QuSpinError::NonConvergence {
            iterations: size,
            residual: failure_residual,
        });
    }
    Ok(Eigensystem {
        eigenvalues: candidates.iter().map(|candidate| candidate.0).collect(),
        eigenvectors: candidates
            .into_iter()
            .map(|candidate| candidate.1)
            .collect(),
        residuals,
        iterations: size,
    })
}

/// Selected Hermitian eigenpairs.
///
/// Small problems use a dense real-symmetric decomposition. Larger problems
/// use a matrix-free, fully reorthogonalized Lanczos backend; shift targets
/// apply a restarted GMRES inverse without materializing the operator.
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
    if shape.0 > 128 {
        return lanczos_eigsh(operator, &options);
    }
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

struct LanczosProjection {
    initial_norm: f64,
    basis: Vec<Vec<Complex64>>,
    eigenvalues: Vec<f64>,
    eigenvectors: DMatrix<f64>,
}

fn lanczos_projection(
    operator: &(impl LinearOperator + ?Sized),
    initial: &[Complex64],
    dimension: usize,
) -> Result<LanczosProjection> {
    let initial_norm = vector_norm(initial);
    if initial_norm <= f64::EPSILON {
        return Ok(LanczosProjection {
            initial_norm,
            basis: Vec::new(),
            eigenvalues: Vec::new(),
            eigenvectors: DMatrix::zeros(0, 0),
        });
    }
    let krylov_dimension = dimension.min(initial.len()).max(1);
    let mut first = initial.to_vec();
    for value in &mut first {
        *value /= initial_norm;
    }
    let mut basis = Vec::with_capacity(krylov_dimension);
    basis.push(first);
    let mut alphas = Vec::with_capacity(krylov_dimension);
    let mut betas = Vec::with_capacity(krylov_dimension.saturating_sub(1));
    let mut applied = vec![Complex64::new(0.0, 0.0); initial.len()];

    for iteration in 0..krylov_dimension {
        operator.apply(&basis[iteration], &mut applied)?;
        let alpha = inner(&basis[iteration], &applied);
        if alpha.im.abs() > 1.0e-10 {
            return Err(QuSpinError::NonHermitian);
        }
        alphas.push(alpha.re);
        for (value, basis_value) in applied.iter_mut().zip(&basis[iteration]) {
            *value -= alpha.re * *basis_value;
        }
        if iteration > 0 {
            for (value, previous) in applied.iter_mut().zip(&basis[iteration - 1]) {
                *value -= betas[iteration - 1] * *previous;
            }
        }
        // Exponential actions use the Hermitian three-term recurrence. Unlike
        // the multi-eigenpair solver, they do not need global Ritz-vector
        // orthogonality, so avoiding O(m^2 n) reorthogonalization is essential
        // for the 100-step paper workflows.
        let beta = vector_norm(&applied);
        if iteration + 1 == krylov_dimension || beta <= 1.0e-14 {
            break;
        }
        betas.push(beta);
        for value in &mut applied {
            *value /= beta;
        }
        basis.push(applied.clone());
    }

    let size = basis.len();
    let mut tridiagonal = DMatrix::<f64>::zeros(size, size);
    for index in 0..size {
        tridiagonal[(index, index)] = alphas[index];
        if index + 1 < size {
            tridiagonal[(index, index + 1)] = betas[index];
            tridiagonal[(index + 1, index)] = betas[index];
        }
    }
    let decomposition = SymmetricEigen::new(tridiagonal);
    Ok(LanczosProjection {
        initial_norm,
        basis,
        eigenvalues: decomposition.eigenvalues.as_slice().to_vec(),
        eigenvectors: decomposition.eigenvectors,
    })
}

fn projected_exponential_action(
    projection: &LanczosProjection,
    interval: f64,
    hamiltonian: bool,
    ambient_dimension: usize,
) -> Vec<Complex64> {
    if projection.basis.is_empty() {
        return vec![Complex64::new(0.0, 0.0); ambient_dimension];
    }
    let size = projection.basis.len();
    let mut coefficients = vec![Complex64::new(0.0, 0.0); size];
    for eigen_index in 0..size {
        let exponent = if hamiltonian {
            Complex64::new(0.0, -interval * projection.eigenvalues[eigen_index]).exp()
        } else {
            Complex64::new(interval * projection.eigenvalues[eigen_index], 0.0).exp()
        };
        let weight = projection.initial_norm * projection.eigenvectors[(0, eigen_index)] * exponent;
        for (basis_index, coefficient) in coefficients.iter_mut().enumerate() {
            *coefficient += projection.eigenvectors[(basis_index, eigen_index)] * weight;
        }
    }
    let mut output = vec![Complex64::new(0.0, 0.0); ambient_dimension];
    for (coefficient, vector) in coefficients.iter().zip(&projection.basis) {
        for (value, basis_value) in output.iter_mut().zip(vector) {
            *value += *coefficient * *basis_value;
        }
    }
    if hamiltonian {
        let output_norm = vector_norm(&output);
        if output_norm > f64::EPSILON && output_norm.is_finite() {
            let scale = projection.initial_norm / output_norm;
            for value in &mut output {
                *value *= scale;
            }
        }
    }
    output
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
    if options.hamiltonian {
        let projection = lanczos_projection(operator, initial, options.krylov_dimension)?;
        return Ok(projected_exponential_action(
            &projection,
            interval,
            true,
            initial.len(),
        ));
    }
    let requested_steps = interval.abs().ceil().max(1.0) as usize;
    if requested_steps > options.max_substeps {
        return Err(QuSpinError::NonConvergence {
            iterations: options.max_substeps,
            residual: interval.abs(),
        });
    }
    let step = interval / requested_steps as f64;
    let factor = Complex64::new(step, 0.0);
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
    if options.hamiltonian {
        let projection = lanczos_projection(operator, initial, options.krylov_dimension)?;
        for &time in &options.times {
            states.push(projected_exponential_action(
                &projection,
                time,
                true,
                initial.len(),
            ));
        }
        return Ok(StateTrajectory {
            times: options.times,
            states,
        });
    }
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
