use std::sync::Arc;

use num_complex::Complex64;

use crate::operator::{
    LinearOperator, MatrixFormat, Operator, TimeDependentOperator, check_apply_shape,
};
use crate::{QmbedError, Result};

fn block_offsets(shapes: impl IntoIterator<Item = (usize, usize)>) -> Result<Vec<usize>> {
    let mut offsets = vec![0_usize];
    for shape in shapes {
        if shape.0 != shape.1 {
            return Err(QmbedError::DimensionMismatch(
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
                    QmbedError::UnsupportedBackend("block dimension overflow".into())
                })?,
        );
    }
    if offsets.len() == 1 {
        return Err(QmbedError::InvalidOptions(
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

    pub fn push(&mut self, block: Arc<dyn LinearOperator>) -> Result<()> {
        if block.shape().0 != block.shape().1 {
            return Err(QmbedError::DimensionMismatch(
                "block-diagonal operators require square blocks".into(),
            ));
        }
        let next = self
            .offsets
            .last()
            .copied()
            .unwrap_or_default()
            .checked_add(block.shape().0)
            .ok_or_else(|| QmbedError::UnsupportedBackend("block dimension overflow".into()))?;
        self.blocks.push(block);
        self.offsets.push(next);
        Ok(())
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

    fn stored_triplets(&self) -> Result<Option<Vec<(usize, usize, Complex64)>>> {
        let mut entries = Vec::new();
        for (block_index, block) in self.blocks.iter().enumerate() {
            let Some(block_entries) = block.stored_triplets()? else {
                return Ok(None);
            };
            let offset = self.offsets[block_index];
            entries.extend(
                block_entries
                    .into_iter()
                    .map(|(row, column, value)| (offset + row, offset + column, value)),
            );
        }
        Ok(Some(entries))
    }
}

fn streamed_triplets(
    operator: &(impl LinearOperator + ?Sized),
) -> Result<Vec<(usize, usize, Complex64)>> {
    if let Some(entries) = operator.stored_triplets()? {
        return Ok(entries);
    }
    let shape = operator.shape();
    let mut input = vec![Complex64::new(0.0, 0.0); shape.1];
    let mut output = vec![Complex64::new(0.0, 0.0); shape.0];
    let mut entries = Vec::new();
    for column in 0..shape.1 {
        input.fill(Complex64::new(0.0, 0.0));
        input[column] = Complex64::new(1.0, 0.0);
        operator.apply(&input, &mut output)?;
        for (row, value) in output.iter().copied().enumerate() {
            if value.norm() > f64::EPSILON {
                entries.push((row, column, value));
            }
        }
    }
    Ok(entries)
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
        let offset = offsets[block_index];
        triplets.extend(
            streamed_triplets(block.as_ref())?
                .into_iter()
                .map(|(row, column, value)| (offset + row, offset + column, value)),
        );
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

    pub fn push(&mut self, block: Arc<dyn TimeDependentOperator>) -> Result<()> {
        if block.shape().0 != block.shape().1 {
            return Err(QmbedError::DimensionMismatch(
                "block-diagonal operators require square blocks".into(),
            ));
        }
        let next = self
            .offsets
            .last()
            .copied()
            .unwrap_or_default()
            .checked_add(block.shape().0)
            .ok_or_else(|| QmbedError::UnsupportedBackend("block dimension overflow".into()))?;
        self.blocks.push(block);
        self.offsets.push(next);
        Ok(())
    }

    pub fn materialize(&self, time: f64, format: MatrixFormat) -> Result<Operator> {
        if !time.is_finite() {
            return Err(QmbedError::InvalidOptions(
                "dynamic block materialization time must be finite".into(),
            ));
        }
        let dimension = self.offsets.last().copied().unwrap_or_default();
        let mut entries = Vec::new();
        for (block_index, block) in self.blocks.iter().enumerate() {
            let block_dimension = block.shape().0;
            let offset = self.offsets[block_index];
            let mut input = vec![Complex64::new(0.0, 0.0); block_dimension];
            let mut output = vec![Complex64::new(0.0, 0.0); block_dimension];
            for column in 0..block_dimension {
                input.fill(Complex64::new(0.0, 0.0));
                input[column] = Complex64::new(1.0, 0.0);
                block.apply_at(time, &input, &mut output)?;
                for (row, value) in output.iter().copied().enumerate() {
                    if value.norm() > f64::EPSILON {
                        entries.push((offset + row, offset + column, value));
                    }
                }
            }
        }
        Operator::from_triplets(dimension, dimension, entries, format)
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
