use std::sync::Arc;

use num_complex::Complex64;

use crate::operator::{
    LinearOperator, MatrixFormat, Operator, TimeDependentOperator, check_apply_shape,
    materialize_dense,
};
use crate::{QuSpinError, Result};

fn block_offsets(shapes: impl IntoIterator<Item = (usize, usize)>) -> Result<Vec<usize>> {
    let mut offsets = vec![0_usize];
    for shape in shapes {
        if shape.0 != shape.1 {
            return Err(QuSpinError::DimensionMismatch(
                "block-diagonal operators require square blocks".into(),
            ));
        }
        offsets.push(
            offsets
                .last()
                .copied()
                .unwrap_or_default()
                .checked_add(shape.0)
                .ok_or_else(|| {
                    QuSpinError::UnsupportedBackend("block dimension overflow".into())
                })?,
        );
    }
    if offsets.len() == 1 {
        return Err(QuSpinError::InvalidOptions(
            "at least one operator block is required".into(),
        ));
    }
    Ok(offsets)
}

/// Delayed matrix-free direct sum of static blocks.
pub struct BlockOps {
    blocks: Vec<Arc<dyn LinearOperator>>,
    offsets: Vec<usize>,
}

impl BlockOps {
    pub fn new(blocks: impl IntoIterator<Item = Arc<dyn LinearOperator>>) -> Result<Self> {
        let blocks: Vec<_> = blocks.into_iter().collect();
        let offsets = block_offsets(blocks.iter().map(|block| block.shape()))?;
        Ok(Self { blocks, offsets })
    }

    pub fn blocks(&self) -> usize {
        self.blocks.len()
    }

    pub fn materialize(&self, format: MatrixFormat) -> Result<Operator> {
        block_diag_hamiltonian(self.blocks.iter().cloned(), format)
    }
}

impl LinearOperator for BlockOps {
    fn shape(&self) -> (usize, usize) {
        let dimension = self.offsets.last().copied().unwrap_or_default();
        (dimension, dimension)
    }

    fn format(&self) -> MatrixFormat {
        MatrixFormat::MatrixFree
    }

    fn apply(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        check_apply_shape(self.shape(), input, output)?;
        output.fill(Complex64::new(0.0, 0.0));
        for (index, block) in self.blocks.iter().enumerate() {
            let start = self.offsets[index];
            let end = self.offsets[index + 1];
            block.apply(&input[start..end], &mut output[start..end])?;
        }
        Ok(())
    }
}

pub fn block_diag_hamiltonian(
    blocks: impl IntoIterator<Item = Arc<dyn LinearOperator>>,
    format: MatrixFormat,
) -> Result<Operator> {
    let blocks: Vec<_> = blocks.into_iter().collect();
    let offsets = block_offsets(blocks.iter().map(|block| block.shape()))?;
    let dimension = offsets.last().copied().unwrap_or_default();
    let mut triplets = Vec::new();
    for (block_index, block) in blocks.iter().enumerate() {
        let block_dimension = block.shape().0;
        let dense = materialize_dense(block.as_ref())?;
        let offset = offsets[block_index];
        for row in 0..block_dimension {
            for column in 0..block_dimension {
                let value = dense[row * block_dimension + column];
                if value.norm() > f64::EPSILON {
                    triplets.push((offset + row, offset + column, value));
                }
            }
        }
    }
    Operator::from_triplets(dimension, dimension, triplets, format)
}

/// Delayed direct sum of explicitly time-dependent blocks.
pub struct DynamicBlockOps {
    blocks: Vec<Arc<dyn TimeDependentOperator>>,
    offsets: Vec<usize>,
}

impl DynamicBlockOps {
    pub fn new(blocks: impl IntoIterator<Item = Arc<dyn TimeDependentOperator>>) -> Result<Self> {
        let blocks: Vec<_> = blocks.into_iter().collect();
        let offsets = block_offsets(blocks.iter().map(|block| block.shape()))?;
        Ok(Self { blocks, offsets })
    }

    pub fn blocks(&self) -> usize {
        self.blocks.len()
    }
}

impl TimeDependentOperator for DynamicBlockOps {
    fn shape(&self) -> (usize, usize) {
        let dimension = self.offsets.last().copied().unwrap_or_default();
        (dimension, dimension)
    }

    fn apply_at(&self, time: f64, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        check_apply_shape(self.shape(), input, output)?;
        output.fill(Complex64::new(0.0, 0.0));
        for (index, block) in self.blocks.iter().enumerate() {
            let start = self.offsets[index];
            let end = self.offsets[index + 1];
            block.apply_at(time, &input[start..end], &mut output[start..end])?;
        }
        Ok(())
    }
}
