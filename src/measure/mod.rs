use num_complex::Complex64;

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
        let mut columns = Vec::with_capacity(rank);
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
