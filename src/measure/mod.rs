use std::collections::HashMap;

use num_complex::Complex64;

use crate::basis::BasisProjector;
use crate::operator::{LinearOperator, MatrixFormat, Operator};
use crate::solve::{StateTrajectory, eigh};
use crate::{QuSpinError, Result};

/// Gauge-independent finite-dimensional subspace.
#[derive(Clone, Debug)]
pub struct Subspace {
    ambient_dimension: usize,
    columns: Vec<Vec<Complex64>>,
}

impl Subspace {
    pub fn from_columns(
        ambient_dimension: usize,
        rank: usize,
        column_major_vectors: Vec<Complex64>,
    ) -> Result<Self> {
        if ambient_dimension == 0
            || rank == 0
            || column_major_vectors.len() != ambient_dimension.saturating_mul(rank)
        {
            return Err(QuSpinError::DimensionMismatch(
                "subspace storage must contain ambient_dimension * rank entries".into(),
            ));
        }
        let mut columns: Vec<Vec<Complex64>> = Vec::with_capacity(rank);
        for column in 0..rank {
            let mut vector = column_major_vectors
                [column * ambient_dimension..(column + 1) * ambient_dimension]
                .to_vec();
            for previous in &columns {
                let overlap = inner(previous, &vector);
                for (value, basis_value) in vector.iter_mut().zip(previous) {
                    *value -= overlap * *basis_value;
                }
            }
            let norm = vector.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
            if norm <= 1.0e-13 {
                return Err(QuSpinError::RankDeficient);
            }
            for value in &mut vector {
                *value /= norm;
            }
            columns.push(vector);
        }
        Ok(Self {
            ambient_dimension,
            columns,
        })
    }

    pub const fn ambient_dimension(&self) -> usize {
        self.ambient_dimension
    }

    pub fn rank(&self) -> usize {
        self.columns.len()
    }

    pub fn columns(&self) -> &[Vec<Complex64>] {
        &self.columns
    }
}

fn inner(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    left.iter()
        .zip(right)
        .map(|(left_value, right_value)| left_value.conj() * *right_value)
        .sum()
}

/// Mean squared principal-angle cosine between two subspaces.
pub fn subspace_fidelity(left: &Subspace, right: &Subspace) -> Result<f64> {
    if left.ambient_dimension != right.ambient_dimension {
        return Err(QuSpinError::DimensionMismatch(
            "subspaces must share an ambient dimension".into(),
        ));
    }
    let denominator = left.rank().min(right.rank());
    if denominator == 0 {
        return Err(QuSpinError::RankDeficient);
    }
    let overlap_norm: f64 = left
        .columns
        .iter()
        .flat_map(|left_vector| {
            right
                .columns
                .iter()
                .map(move |right_vector| inner(left_vector, right_vector).norm_sqr())
        })
        .sum();
    Ok((overlap_norm / denominator as f64).clamp(0.0, 1.0))
}

pub fn matrix_element(
    left: &[Complex64],
    operator: &(impl LinearOperator + ?Sized),
    right: &[Complex64],
) -> Result<Complex64> {
    let shape = operator.shape();
    if left.len() != shape.0 || right.len() != shape.1 {
        return Err(QuSpinError::DimensionMismatch(
            "matrix-element vectors do not match the operator shape".into(),
        ));
    }
    let mut applied = vec![Complex64::new(0.0, 0.0); shape.0];
    operator.apply(right, &mut applied)?;
    Ok(inner(left, &applied))
}

pub fn expectation(
    operator: &(impl LinearOperator + ?Sized),
    state: &[Complex64],
) -> Result<Complex64> {
    matrix_element(state, operator, state)
}

/// Variance `||A psi||^2 - |<psi|A|psi>|^2` for a normalized state.
pub fn quantum_fluctuation(
    operator: &(impl LinearOperator + ?Sized),
    state: &[Complex64],
) -> Result<f64> {
    let shape = operator.shape();
    if shape.0 != shape.1 || state.len() != shape.0 {
        return Err(QuSpinError::DimensionMismatch(
            "quantum fluctuation requires a square operator matching the state".into(),
        ));
    }
    let norm = inner(state, state).re;
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(QuSpinError::InvalidOptions(
            "state must have positive finite norm".into(),
        ));
    }
    let mut applied = vec![Complex64::new(0.0, 0.0); state.len()];
    operator.apply(state, &mut applied)?;
    let mean = inner(state, &applied) / norm;
    let second = inner(&applied, &applied).re / norm;
    Ok((second - mean.norm_sqr()).max(0.0))
}

/// Reduced density matrix of the first factor of a pure bipartite state.
pub fn partial_trace(
    state: &[Complex64],
    subsystem_dimension: usize,
    environment_dimension: usize,
) -> Result<Vec<Complex64>> {
    if subsystem_dimension == 0
        || environment_dimension == 0
        || state.len()
            != subsystem_dimension
                .checked_mul(environment_dimension)
                .ok_or_else(|| {
                    QuSpinError::DimensionMismatch("tensor-product dimension overflow".into())
                })?
    {
        return Err(QuSpinError::DimensionMismatch(
            "state length must equal subsystem_dimension * environment_dimension".into(),
        ));
    }
    let norm = state.iter().map(Complex64::norm_sqr).sum::<f64>();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(QuSpinError::InvalidOptions(
            "state must have positive finite norm".into(),
        ));
    }
    let mut density = vec![Complex64::new(0.0, 0.0); subsystem_dimension * subsystem_dimension];
    for left in 0..subsystem_dimension {
        for right in 0..subsystem_dimension {
            density[left * subsystem_dimension + right] = (0..environment_dimension)
                .map(|environment| {
                    state[left * environment_dimension + environment]
                        * state[right * environment_dimension + environment].conj()
                        / norm
                })
                .sum();
        }
    }
    Ok(density)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EntropyOrder {
    VonNeumann,
    Renyi(f64),
}

pub fn entanglement_entropy(
    state: &[Complex64],
    subsystem_dimension: usize,
    environment_dimension: usize,
    order: EntropyOrder,
) -> Result<f64> {
    if matches!(order, EntropyOrder::Renyi(alpha) if !alpha.is_finite() || alpha <= 0.0 || (alpha - 1.0).abs() <= 1.0e-12)
    {
        return Err(QuSpinError::InvalidOptions(
            "Renyi order must be positive, finite, and different from one".into(),
        ));
    }
    let density = partial_trace(state, subsystem_dimension, environment_dimension)?;
    let operator = Operator::from_dense(subsystem_dimension, subsystem_dimension, density)?;
    let spectrum = eigh(&operator)?;
    let probabilities: Vec<_> = spectrum
        .eigenvalues
        .into_iter()
        .map(|value| {
            if value < -1.0e-10 {
                Err(QuSpinError::InvalidOptions(
                    "reduced density matrix is not positive".into(),
                ))
            } else {
                Ok(value.max(0.0))
            }
        })
        .collect::<Result<_>>()?;
    match order {
        EntropyOrder::VonNeumann => Ok(-probabilities
            .into_iter()
            .filter(|probability| *probability > f64::EPSILON)
            .map(|probability| probability * probability.ln())
            .sum::<f64>()),
        EntropyOrder::Renyi(alpha) => Ok(probabilities
            .into_iter()
            .map(|probability| probability.powf(alpha))
            .sum::<f64>()
            .ln()
            / (1.0 - alpha)),
    }
}

pub fn observables_vs_time(
    trajectory: &StateTrajectory,
    observables: &[(String, &dyn LinearOperator)],
) -> Result<HashMap<String, Vec<Complex64>>> {
    if trajectory.times.len() != trajectory.states.len() {
        return Err(QuSpinError::DimensionMismatch(
            "trajectory times and states must have equal lengths".into(),
        ));
    }
    let mut result = HashMap::with_capacity(observables.len());
    for (name, operator) in observables {
        if name.is_empty() || result.contains_key(name) {
            return Err(QuSpinError::InvalidOptions(
                "observable names must be nonempty and unique".into(),
            ));
        }
        let values = trajectory
            .states
            .iter()
            .map(|state| expectation(*operator, state))
            .collect::<Result<_>>()?;
        result.insert(name.clone(), values);
    }
    Ok(result)
}

/// Exact time evolution from a complete eigendecomposition.
pub fn ed_state_vs_time(
    initial: &[Complex64],
    eigenvalues: &[f64],
    eigenvectors: &[Vec<Complex64>],
    times: &[f64],
) -> Result<StateTrajectory> {
    let dimension = initial.len();
    if times.is_empty()
        || times.iter().any(|time| !time.is_finite())
        || eigenvalues.len() != dimension
        || eigenvectors.len() != dimension
        || eigenvectors.iter().any(|vector| vector.len() != dimension)
    {
        return Err(QuSpinError::DimensionMismatch(
            "complete eigensystem, state, and finite nonempty times are required".into(),
        ));
    }
    let coefficients: Vec<_> = eigenvectors
        .iter()
        .map(|vector| inner(vector, initial))
        .collect();
    let states = times
        .iter()
        .map(|time| {
            let mut state = vec![Complex64::new(0.0, 0.0); dimension];
            for ((energy, vector), coefficient) in
                eigenvalues.iter().zip(eigenvectors).zip(&coefficients)
            {
                let weight = coefficient * Complex64::new(0.0, -*time * energy).exp();
                for (value, eigenvector_value) in state.iter_mut().zip(vector) {
                    *value += weight * *eigenvector_value;
                }
            }
            state
        })
        .collect();
    Ok(StateTrajectory {
        times: times.to_vec(),
        states,
    })
}

/// Density-matrix counterpart of [`ed_state_vs_time`], with row-major input
/// and output matrices.
pub fn ed_density_vs_time(
    initial: &[Complex64],
    eigenvalues: &[f64],
    eigenvectors: &[Vec<Complex64>],
    times: &[f64],
) -> Result<Vec<Vec<Complex64>>> {
    let dimension = eigenvalues.len();
    if initial.len() != dimension.saturating_mul(dimension)
        || eigenvectors.len() != dimension
        || eigenvectors.iter().any(|vector| vector.len() != dimension)
        || times.is_empty()
        || times.iter().any(|time| !time.is_finite())
    {
        return Err(QuSpinError::DimensionMismatch(
            "density matrix and complete eigensystem dimensions do not match".into(),
        ));
    }
    let mut eigen_density = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    for left in 0..dimension {
        for right in 0..dimension {
            for row in 0..dimension {
                for column in 0..dimension {
                    eigen_density[left * dimension + right] += eigenvectors[left][row].conj()
                        * initial[row * dimension + column]
                        * eigenvectors[right][column];
                }
            }
        }
    }
    Ok(times
        .iter()
        .map(|time| {
            let mut density = vec![Complex64::new(0.0, 0.0); dimension * dimension];
            for row in 0..dimension {
                for column in 0..dimension {
                    for left in 0..dimension {
                        for right in 0..dimension {
                            let phase = Complex64::new(
                                0.0,
                                -*time * (eigenvalues[left] - eigenvalues[right]),
                            )
                            .exp();
                            density[row * dimension + column] += eigenvectors[left][row]
                                * phase
                                * eigen_density[left * dimension + right]
                                * eigenvectors[right][column].conj();
                        }
                    }
                }
            }
            density
        })
        .collect())
}

#[derive(Clone, Debug)]
pub struct DiagonalEnsemble {
    pub probabilities: Vec<f64>,
    pub mean_energy: f64,
    pub energy_variance: f64,
    pub entropy: f64,
}

pub fn diagonal_ensemble(
    eigenvalues: &[f64],
    eigenvectors: &[Vec<Complex64>],
    initial: &[Complex64],
) -> Result<DiagonalEnsemble> {
    if eigenvalues.len() != eigenvectors.len()
        || eigenvectors
            .iter()
            .any(|vector| vector.len() != initial.len())
    {
        return Err(QuSpinError::DimensionMismatch(
            "eigensystem and initial state dimensions do not match".into(),
        ));
    }
    let initial_norm = inner(initial, initial).re;
    if !initial_norm.is_finite() || initial_norm <= f64::EPSILON {
        return Err(QuSpinError::InvalidOptions(
            "initial state must have positive finite norm".into(),
        ));
    }
    let mut probabilities: Vec<_> = eigenvectors
        .iter()
        .map(|vector| inner(vector, initial).norm_sqr() / initial_norm)
        .collect();
    let probability_sum = probabilities.iter().sum::<f64>();
    if probability_sum <= f64::EPSILON {
        return Err(QuSpinError::InvalidOptions(
            "eigenvectors have no overlap with the initial state".into(),
        ));
    }
    for probability in &mut probabilities {
        *probability /= probability_sum;
    }
    let mean_energy = probabilities
        .iter()
        .zip(eigenvalues)
        .map(|(probability, energy)| probability * energy)
        .sum::<f64>();
    let energy_variance = probabilities
        .iter()
        .zip(eigenvalues)
        .map(|(probability, energy)| probability * (energy - mean_energy).powi(2))
        .sum();
    let entropy = -probabilities
        .iter()
        .filter(|probability| **probability > f64::EPSILON)
        .map(|probability| probability * probability.ln())
        .sum::<f64>();
    Ok(DiagonalEnsemble {
        probabilities,
        mean_energy,
        energy_variance,
        entropy,
    })
}

pub fn kl_divergence(left: &[f64], right: &[f64]) -> Result<f64> {
    if left.len() != right.len()
        || left.is_empty()
        || left
            .iter()
            .chain(right)
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(QuSpinError::InvalidOptions(
            "KL distributions must be nonempty, equal-length, finite, and strictly positive".into(),
        ));
    }
    let left_sum = left.iter().sum::<f64>();
    let right_sum = right.iter().sum::<f64>();
    if (left_sum - 1.0).abs() > 1.0e-13 || (right_sum - 1.0).abs() > 1.0e-13 {
        return Err(QuSpinError::InvalidOptions(
            "KL distributions must be normalized".into(),
        ));
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(probability, reference)| probability * (probability / reference).ln())
        .sum::<f64>()
        .max(0.0))
}

/// Mean adjacent-gap ratio of an ordered spectrum.
pub fn mean_level_spacing(eigenvalues: &[f64]) -> Result<f64> {
    if eigenvalues.len() < 3 || eigenvalues.iter().any(|value| !value.is_finite()) {
        return Err(QuSpinError::InvalidOptions(
            "level-spacing statistics require at least three finite values".into(),
        ));
    }
    if eigenvalues.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(QuSpinError::InvalidOptions(
            "level spectrum must be sorted in ascending order".into(),
        ));
    }
    let gaps: Vec<_> = eigenvalues
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect();
    if gaps.iter().any(|gap| *gap == 0.0) {
        return Ok(f64::NAN);
    }
    Ok(gaps
        .windows(2)
        .map(|pair| pair[0].min(pair[1]) / pair[0].max(pair[1]))
        .sum::<f64>()
        / (gaps.len() - 1) as f64)
}

pub fn states_to_array(
    states: &[u128],
    sites: usize,
    local_dimension: usize,
) -> Result<Vec<Vec<usize>>> {
    if local_dimension < 2 {
        return Err(QuSpinError::InvalidSector(
            "local dimension must be at least two".into(),
        ));
    }
    let base = local_dimension as u128;
    let mut result = Vec::with_capacity(states.len());
    for &state in states {
        let mut value = state;
        let mut occupations = Vec::with_capacity(sites);
        for _ in 0..sites {
            occupations.push((value % base) as usize);
            value /= base;
        }
        if value != 0 {
            return Err(QuSpinError::StateNotInBasis);
        }
        result.push(occupations);
    }
    Ok(result)
}

pub fn array_to_states(arrays: &[Vec<usize>], local_dimension: usize) -> Result<Vec<u128>> {
    if local_dimension < 2 {
        return Err(QuSpinError::InvalidSector(
            "local dimension must be at least two".into(),
        ));
    }
    let sites = arrays.first().map_or(0, Vec::len);
    let base = local_dimension as u128;
    arrays
        .iter()
        .map(|occupations| {
            if occupations.len() != sites
                || occupations
                    .iter()
                    .any(|occupation| *occupation >= local_dimension)
            {
                return Err(QuSpinError::InvalidSector(
                    "occupation arrays must have equal length and valid digits".into(),
                ));
            }
            let mut state = 0_u128;
            let mut place = 1_u128;
            for &occupation in occupations {
                state = state
                    .checked_add(place.checked_mul(occupation as u128).ok_or_else(|| {
                        QuSpinError::UnsupportedBackend("state encoding overflow".into())
                    })?)
                    .ok_or_else(|| {
                        QuSpinError::UnsupportedBackend("state encoding overflow".into())
                    })?;
                place = place.checked_mul(base).ok_or_else(|| {
                    QuSpinError::UnsupportedBackend("state encoding overflow".into())
                })?;
            }
            Ok(state)
        })
        .collect()
}

/// Convert integer states to most-significant-site-first binary rows.
pub fn ints_to_array(states: &[u128], sites: usize) -> Result<Vec<Vec<u8>>> {
    if sites > 128 {
        return Err(QuSpinError::UnsupportedBackend(
            "u128 binary conversion supports at most 128 sites".into(),
        ));
    }
    if states
        .iter()
        .any(|state| sites < 128 && *state >= (1_u128 << sites))
    {
        return Err(QuSpinError::StateNotInBasis);
    }
    Ok(states
        .iter()
        .map(|state| {
            (0..sites)
                .map(|column| ((state >> (sites - column - 1)) & 1) as u8)
                .collect()
        })
        .collect())
}

/// Convert most-significant-site-first binary rows to integer states.
pub fn array_to_ints(arrays: &[Vec<u8>]) -> Result<Vec<u128>> {
    let sites = arrays.first().map_or(0, Vec::len);
    if sites > 128 {
        return Err(QuSpinError::UnsupportedBackend(
            "u128 binary conversion supports at most 128 sites".into(),
        ));
    }
    arrays
        .iter()
        .map(|row| {
            if row.len() != sites || row.iter().any(|bit| *bit > 1) {
                return Err(QuSpinError::InvalidSector(
                    "binary state rows must have equal lengths and contain only zero or one".into(),
                ));
            }
            Ok(row
                .iter()
                .fold(0_u128, |state, bit| (state << 1) | u128::from(*bit)))
        })
        .collect()
}

/// Compute `P† A P` one reduced column at a time.
pub fn project_operator(
    operator: &(impl LinearOperator + ?Sized),
    projector: &BasisProjector,
    format: MatrixFormat,
) -> Result<Operator> {
    let source_dimension = projector.source_dimension();
    let reduced_dimension = projector.reduced_dimension();
    let operator_dimension = operator.shape();
    if operator_dimension.0 != operator_dimension.1
        || (operator_dimension.0 != source_dimension && operator_dimension.0 != reduced_dimension)
    {
        return Err(QuSpinError::DimensionMismatch(
            "operator dimension must match the parent or reduced projector space".into(),
        ));
    }
    if operator_dimension.0 == reduced_dimension {
        let mut parent_input = vec![Complex64::new(0.0, 0.0); source_dimension];
        let mut reduced_input = vec![Complex64::new(0.0, 0.0); reduced_dimension];
        let mut reduced_output = vec![Complex64::new(0.0, 0.0); reduced_dimension];
        let mut parent_output = vec![Complex64::new(0.0, 0.0); source_dimension];
        let mut triplets = Vec::new();
        for column in 0..source_dimension {
            parent_input.fill(Complex64::new(0.0, 0.0));
            parent_input[column] = Complex64::new(1.0, 0.0);
            projector.project(&parent_input, &mut reduced_input)?;
            operator.apply(&reduced_input, &mut reduced_output)?;
            projector.apply(&reduced_output, &mut parent_output)?;
            for (row, &value) in parent_output.iter().enumerate() {
                if value.norm() > f64::EPSILON {
                    triplets.push((row, column, value));
                }
            }
        }
        return Operator::from_triplets(source_dimension, source_dimension, triplets, format);
    }
    let mut reduced_input = vec![Complex64::new(0.0, 0.0); reduced_dimension];
    let mut parent_input = vec![Complex64::new(0.0, 0.0); source_dimension];
    let mut parent_output = vec![Complex64::new(0.0, 0.0); source_dimension];
    let mut reduced_output = vec![Complex64::new(0.0, 0.0); reduced_dimension];
    let mut triplets = Vec::new();
    for column in 0..reduced_dimension {
        reduced_input.fill(Complex64::new(0.0, 0.0));
        reduced_input[column] = Complex64::new(1.0, 0.0);
        projector.apply(&reduced_input, &mut parent_input)?;
        operator.apply(&parent_input, &mut parent_output)?;
        projector.project(&parent_output, &mut reduced_output)?;
        for (row, &value) in reduced_output.iter().enumerate() {
            if value.norm() > f64::EPSILON {
                triplets.push((row, column, value));
            }
        }
    }
    Operator::from_triplets(reduced_dimension, reduced_dimension, triplets, format)
}
