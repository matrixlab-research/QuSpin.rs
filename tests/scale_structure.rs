use std::sync::Arc;

use quspin::basis::{Basis, BosonBasis1D, SpinBasis1D, SpinlessFermionBasis1D};
use quspin::block::block_diag_hamiltonian;
use quspin::operator::{LinearOperator, MatrixFormat, Operator};
use quspin::{Complex64, QuSpinError, Result};

#[derive(Debug)]
struct StoredOnly {
    dimension: usize,
}

impl LinearOperator for StoredOnly {
    fn shape(&self) -> (usize, usize) {
        (self.dimension, self.dimension)
    }

    fn format(&self) -> MatrixFormat {
        MatrixFormat::Csc
    }

    fn apply(&self, _input: &[Complex64], _output: &mut [Complex64]) -> Result<()> {
        Err(QuSpinError::UnsupportedBackend(
            "column probing must not be used for stored blocks".into(),
        ))
    }

    fn stored_triplets(&self) -> Result<Option<Vec<(usize, usize, Complex64)>>> {
        Ok(Some(
            (0..self.dimension)
                .map(|index| (index, index, Complex64::new(index as f64, 0.0)))
                .collect(),
        ))
    }
}

#[test]
fn sparse_algebra_memory_tracks_nonzeros_not_dimension_squared() {
    let dimension = 16_384;
    let diagonal = Operator::from_triplets(
        dimension,
        dimension,
        (0..dimension).map(|index| (index, index, Complex64::new(2.0, 0.0))),
        MatrixFormat::Csc,
    )
    .unwrap();

    let squared = diagonal.product(&diagonal).unwrap();
    let conjugated = squared.conjugated().unwrap();
    let transposed = conjugated.transpose().unwrap();
    assert_eq!(transposed.format(), MatrixFormat::Csc);
    assert_eq!(transposed.nnz(), dimension);
    assert!(transposed.memory_bytes() < 1_000_000);
    assert_eq!(transposed.diagonal()[123], Complex64::new(4.0, 0.0));
}

#[test]
fn block_assembly_streams_owned_nonzeros_without_column_probing() {
    let first: Arc<dyn LinearOperator> = Arc::new(StoredOnly { dimension: 4 });
    let second: Arc<dyn LinearOperator> = Arc::new(StoredOnly { dimension: 3 });
    let assembled = block_diag_hamiltonian([first, second], MatrixFormat::Csr).unwrap();
    assert_eq!(assembled.shape(), (7, 7));
    assert_eq!(assembled.nnz(), 5);
    assert_eq!(assembled.diagonal()[6], Complex64::new(2.0, 0.0));
}

#[test]
fn fixed_particle_enumeration_scales_with_sector_dimension() {
    let single_spin = SpinBasis1D::builder(96).up(1).build().unwrap();
    assert_eq!(single_spin.len(), 96);
    assert_eq!(single_spin.state(95).unwrap(), 1_u128 << 95);

    let fermions = SpinlessFermionBasis1D::builder(96)
        .particles(2)
        .build()
        .unwrap();
    assert_eq!(fermions.len(), 96 * 95 / 2);
    assert_eq!(fermions.state(0).unwrap(), 0b11);

    let bosons = BosonBasis1D::builder(30, 3).particles(1).build().unwrap();
    assert_eq!(bosons.len(), 30);
}
