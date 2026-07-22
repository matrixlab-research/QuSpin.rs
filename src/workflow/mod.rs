use std::sync::Arc;

use num_complex::Complex64;

use crate::operator::{LinearOperator, MatrixFormat, check_apply_shape, materialize_dense};
use crate::{QuSpinError, Result};

/// Matrix-free Lindblad generator over column-major vectorized density matrices.
pub struct LindbladGenerator {
    hamiltonian: Vec<Complex64>,
    jumps: Vec<LindbladJump>,
    dimension: usize,
}

struct LindbladJump {
    operator: Vec<Complex64>,
    adjoint: Vec<Complex64>,
    product: Vec<Complex64>,
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
        let dimension = shape.0;
        let hamiltonian = materialize_dense(hamiltonian.as_ref())?;
        let jumps = jumps
            .into_iter()
            .map(|jump| {
                let operator = materialize_dense(jump.as_ref())?;
                let adjoint = adjoint(&operator, dimension);
                let product = multiply(&adjoint, &operator, dimension);
                Ok(LindbladJump {
                    operator,
                    adjoint,
                    product,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            hamiltonian,
            jumps,
            dimension,
        })
    }
}

fn multiply(left: &[Complex64], right: &[Complex64], dimension: usize) -> Vec<Complex64> {
    let mut product = vec![Complex64::new(0.0, 0.0); dimension * dimension];
    for row in 0..dimension {
        for middle in 0..dimension {
            for column in 0..dimension {
                product[row * dimension + column] +=
                    left[row * dimension + middle] * right[middle * dimension + column];
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
        let h_rho = multiply(&self.hamiltonian, &density, dimension);
        let rho_h = multiply(&density, &self.hamiltonian, dimension);
        let mut derivative: Vec<_> = h_rho
            .iter()
            .zip(&rho_h)
            .map(|(left, right)| Complex64::new(0.0, -1.0) * (*left - *right))
            .collect();
        for jump in &self.jumps {
            let gain = multiply(
                &multiply(&jump.operator, &density, dimension),
                &jump.adjoint,
                dimension,
            );
            let loss_left = multiply(&jump.product, &density, dimension);
            let loss_right = multiply(&density, &jump.product, dimension);
            for index in 0..derivative.len() {
                derivative[index] += gain[index] - 0.5 * (loss_left[index] + loss_right[index]);
            }
        }
        output.copy_from_slice(&row_major_to_column_major(&derivative, dimension));
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct StateTrackingResult {
    /// `permutation[previous_index]` is the matched current-state index.
    pub permutation: Vec<usize>,
    /// Multiply each matched current state by this phase to align gauges.
    pub phases: Vec<Complex64>,
    pub overlaps: Vec<f64>,
    pub ambiguous: Vec<usize>,
}

fn state_inner(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left.conj() * *right)
        .sum()
}

/// Match two equal-rank eigenvector frames by globally maximizing absolute
/// overlaps, then return gauge-aligning phases and ambiguity diagnostics.
pub fn track_states(
    previous: &[Vec<Complex64>],
    current: &[Vec<Complex64>],
    ambiguity_tolerance: f64,
) -> Result<StateTrackingResult> {
    let rank = previous.len();
    if rank == 0
        || current.len() != rank
        || !ambiguity_tolerance.is_finite()
        || ambiguity_tolerance < 0.0
    {
        return Err(QuSpinError::InvalidOptions(
            "state tracking requires equal nonzero ranks and a nonnegative tolerance".into(),
        ));
    }
    let dimension = previous[0].len();
    if dimension == 0
        || previous.iter().any(|vector| vector.len() != dimension)
        || current.iter().any(|vector| vector.len() != dimension)
    {
        return Err(QuSpinError::DimensionMismatch(
            "tracked state vectors must have equal nonzero dimensions".into(),
        ));
    }
    let overlaps: Vec<Vec<_>> = previous
        .iter()
        .map(|left| {
            current
                .iter()
                .map(|right| state_inner(left, right))
                .collect()
        })
        .collect();

    // Hungarian algorithm for the minimum cost `-abs(overlap)` assignment.
    let mut row_potential = vec![0.0_f64; rank + 1];
    let mut column_potential = vec![0.0_f64; rank + 1];
    let mut matched_row = vec![0_usize; rank + 1];
    let mut predecessor = vec![0_usize; rank + 1];
    for row in 1..=rank {
        matched_row[0] = row;
        let mut column = 0;
        let mut minimum = vec![f64::INFINITY; rank + 1];
        let mut used = vec![false; rank + 1];
        loop {
            used[column] = true;
            let active_row = matched_row[column];
            let mut delta = f64::INFINITY;
            let mut next_column = 0;
            for candidate in 1..=rank {
                if used[candidate] {
                    continue;
                }
                let cost = -overlaps[active_row - 1][candidate - 1].norm()
                    - row_potential[active_row]
                    - column_potential[candidate];
                if cost < minimum[candidate] {
                    minimum[candidate] = cost;
                    predecessor[candidate] = column;
                }
                if minimum[candidate] < delta {
                    delta = minimum[candidate];
                    next_column = candidate;
                }
            }
            for candidate in 0..=rank {
                if used[candidate] {
                    row_potential[matched_row[candidate]] += delta;
                    column_potential[candidate] -= delta;
                } else {
                    minimum[candidate] -= delta;
                }
            }
            column = next_column;
            if matched_row[column] == 0 {
                break;
            }
        }
        loop {
            let previous_column = predecessor[column];
            matched_row[column] = matched_row[previous_column];
            column = previous_column;
            if column == 0 {
                break;
            }
        }
    }
    let mut permutation = vec![0_usize; rank];
    for column in 1..=rank {
        permutation[matched_row[column] - 1] = column - 1;
    }
    let mut phases = Vec::with_capacity(rank);
    let mut assigned_overlaps = Vec::with_capacity(rank);
    let mut ambiguous = Vec::new();
    for row in 0..rank {
        let overlap = overlaps[row][permutation[row]];
        let magnitude = overlap.norm();
        phases.push(if magnitude > f64::EPSILON {
            overlap.conj() / magnitude
        } else {
            Complex64::new(1.0, 0.0)
        });
        assigned_overlaps.push(magnitude);
        let alternative = overlaps[row]
            .iter()
            .enumerate()
            .filter(|(column, _)| *column != permutation[row])
            .map(|(_, value)| value.norm())
            .fold(0.0_f64, f64::max);
        if magnitude - alternative <= ambiguity_tolerance {
            ambiguous.push(row);
        }
    }
    Ok(StateTrackingResult {
        permutation,
        phases,
        overlaps: assigned_overlaps,
        ambiguous,
    })
}
