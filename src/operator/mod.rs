use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;

use faer::linalg::solvers::Solve;
use faer::sparse::{SparseColMat, Triplet as FaerTriplet};
use num_complex::Complex64;

use crate::basis::Basis;
use crate::{QuSpinError, Result};

/// One complex coefficient and its ordered zero-based sites.
#[derive(Clone, Debug, PartialEq)]
pub struct Coupling {
    pub coefficient: Complex64,
    pub sites: Vec<usize>,
}

impl Coupling {
    pub fn new(coefficient: impl Into<Complex64>, sites: impl Into<Vec<usize>>) -> Self {
        Self {
            coefficient: coefficient.into(),
            sites: sites.into(),
        }
    }
}

/// Parsed-once local operator string and its couplings.
#[derive(Clone, Debug, PartialEq)]
pub struct OperatorTerm {
    operator: String,
    couplings: Vec<Coupling>,
}

impl OperatorTerm {
    pub fn new(
        operator: impl AsRef<str>,
        couplings: impl IntoIterator<Item = Coupling>,
    ) -> Result<Self> {
        let operator = operator.as_ref();
        let arity = operator
            .chars()
            .filter(|character| *character != '|')
            .count();
        if arity == 0 {
            return Err(QuSpinError::InvalidOperator(operator.into()));
        }
        let couplings: Vec<_> = couplings.into_iter().collect();
        for coupling in &couplings {
            if coupling.sites.len() != arity {
                return Err(QuSpinError::InvalidCoupling(format!(
                    "operator {operator:?} has arity {arity}, but a coupling has {} sites",
                    coupling.sites.len()
                )));
            }
            if !coupling.coefficient.re.is_finite() || !coupling.coefficient.im.is_finite() {
                return Err(QuSpinError::InvalidCoupling(
                    "coupling coefficients must be finite".into(),
                ));
            }
        }
        Ok(Self {
            operator: operator.into(),
            couplings,
        })
    }

    pub fn operator(&self) -> &str {
        &self.operator
    }

    pub fn couplings(&self) -> &[Coupling] {
        &self.couplings
    }
}

/// Requested materialization backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatrixFormat {
    Dense,
    Csc,
    Csr,
    Dia,
    MatrixFree,
}

/// Rectangular-capable narrow waist shared by stored and matrix-free maps.
pub trait LinearOperator: Send + Sync {
    fn shape(&self) -> (usize, usize);
    fn format(&self) -> MatrixFormat;
    fn apply(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()>;

    fn shifted_solver(&self, _shift: f64) -> Result<Option<Box<dyn ShiftedLinearSolver>>> {
        Ok(None)
    }
}

/// Narrow waist for operators whose action depends explicitly on time.
pub trait TimeDependentOperator: Send + Sync {
    fn shape(&self) -> (usize, usize);
    fn apply_at(&self, time: f64, input: &[Complex64], output: &mut [Complex64]) -> Result<()>;
}

/// Reusable factorization of `(A - shift * I)` for interior eigensolvers.
pub trait ShiftedLinearSolver: Send + Sync {
    fn solve(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AssemblyChecks {
    pub hermiticity: bool,
    pub particle_conservation: bool,
    pub symmetry_compatibility: bool,
}

impl AssemblyChecks {
    pub const fn all() -> Self {
        Self {
            hermiticity: true,
            particle_conservation: true,
            symmetry_compatibility: true,
        }
    }

    pub const fn rectangular() -> Self {
        Self {
            hermiticity: false,
            particle_conservation: false,
            symmetry_compatibility: true,
        }
    }
}

impl Default for AssemblyChecks {
    fn default() -> Self {
        Self::all()
    }
}

#[derive(Clone, Debug)]
struct Triplet {
    row: usize,
    column: usize,
    value: Complex64,
}

#[derive(Clone, Debug)]
enum Storage {
    Dense(Vec<Complex64>),
    Csc {
        column_offsets: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<Complex64>,
    },
    Csr {
        row_offsets: Vec<usize>,
        column_indices: Vec<usize>,
        values: Vec<Complex64>,
    },
    Dia {
        offsets: Vec<isize>,
        values_by_row: Vec<Complex64>,
    },
    MatrixFree(Vec<Triplet>),
}

/// Concrete operator returned by universal assembly.
#[derive(Clone, Debug)]
pub struct Operator {
    shape: (usize, usize),
    format: MatrixFormat,
    storage: Storage,
}

struct FaerShiftedSolver {
    factorization: faer::sparse::linalg::solvers::Lu<usize, Complex64>,
    dimension: usize,
}

impl ShiftedLinearSolver for FaerShiftedSolver {
    fn solve(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        if input.len() != self.dimension || output.len() != self.dimension {
            return Err(QuSpinError::DimensionMismatch(
                "shifted solve input or output length does not match".into(),
            ));
        }
        let mut right_hand_side = faer::Col::from_fn(self.dimension, |index| input[index]);
        self.factorization.solve_in_place(right_hand_side.as_mut());
        for (index, value) in output.iter_mut().enumerate() {
            *value = right_hand_side[index];
        }
        Ok(())
    }
}

impl Operator {
    pub fn from_dense(
        rows: usize,
        columns: usize,
        values_row_major: Vec<Complex64>,
    ) -> Result<Self> {
        if values_row_major.len() != rows.saturating_mul(columns) {
            return Err(QuSpinError::DimensionMismatch(format!(
                "dense storage has {} entries for shape ({rows}, {columns})",
                values_row_major.len()
            )));
        }
        Ok(Self {
            shape: (rows, columns),
            format: MatrixFormat::Dense,
            storage: Storage::Dense(values_row_major),
        })
    }

    pub fn from_triplets(
        rows: usize,
        columns: usize,
        triplets: impl IntoIterator<Item = (usize, usize, Complex64)>,
        format: MatrixFormat,
    ) -> Result<Self> {
        if format == MatrixFormat::Dense {
            let mut values = vec![Complex64::new(0.0, 0.0); rows.saturating_mul(columns)];
            for (row, column, value) in triplets {
                if row >= rows
                    || column >= columns
                    || !value.re.is_finite()
                    || !value.im.is_finite()
                {
                    return Err(QuSpinError::InvalidCoupling(
                        "triplet index is out of bounds or its value is non-finite".into(),
                    ));
                }
                values[row * columns + column] += value;
            }
            return Self::from_dense(rows, columns, values);
        }
        let mut accumulated = HashMap::new();
        for (row, column, value) in triplets {
            if row >= rows || column >= columns || !value.re.is_finite() || !value.im.is_finite() {
                return Err(QuSpinError::InvalidCoupling(
                    "triplet index is out of bounds or its value is non-finite".into(),
                ));
            }
            *accumulated
                .entry((row, column))
                .or_insert(Complex64::new(0.0, 0.0)) += value;
        }
        let mut entries: Vec<_> = accumulated
            .into_iter()
            .filter_map(|((row, column), value)| {
                (value.norm() > f64::EPSILON).then_some(Triplet { row, column, value })
            })
            .collect();
        match format {
            MatrixFormat::Csc => entries.sort_by_key(|entry| (entry.column, entry.row)),
            MatrixFormat::Csr => entries.sort_by_key(|entry| (entry.row, entry.column)),
            MatrixFormat::Dia | MatrixFormat::MatrixFree => {
                entries.sort_by_key(|entry| (entry.row, entry.column));
            }
            MatrixFormat::Dense => unreachable!(),
        }
        let shape = (rows, columns);
        let storage = match format {
            MatrixFormat::Csc => csc_storage(&entries, columns),
            MatrixFormat::Csr => csr_storage(&entries, rows),
            MatrixFormat::Dia => dia_storage(&entries, rows),
            MatrixFormat::MatrixFree => Storage::MatrixFree(entries),
            MatrixFormat::Dense => unreachable!(),
        };
        Ok(Self {
            shape,
            format,
            storage,
        })
    }

    pub fn to_dense(&self) -> Vec<Complex64> {
        if let Storage::Dense(values) = &self.storage {
            return values.clone();
        }
        let mut dense = vec![Complex64::new(0.0, 0.0); self.shape.0 * self.shape.1];
        match &self.storage {
            Storage::Dense(_) => unreachable!(),
            Storage::Csc {
                column_offsets,
                row_indices,
                values,
            } => {
                for column in 0..self.shape.1 {
                    for position in column_offsets[column]..column_offsets[column + 1] {
                        dense[row_indices[position] * self.shape.1 + column] = values[position];
                    }
                }
            }
            Storage::Csr {
                row_offsets,
                column_indices,
                values,
            } => {
                for row in 0..self.shape.0 {
                    for position in row_offsets[row]..row_offsets[row + 1] {
                        dense[row * self.shape.1 + column_indices[position]] = values[position];
                    }
                }
            }
            Storage::Dia {
                offsets,
                values_by_row,
            } => {
                for (diagonal, &offset) in offsets.iter().enumerate() {
                    for row in 0..self.shape.0 {
                        let Some(column) = row.checked_add_signed(-offset) else {
                            continue;
                        };
                        if column < self.shape.1 {
                            dense[row * self.shape.1 + column] =
                                values_by_row[diagonal * self.shape.0 + row];
                        }
                    }
                }
            }
            Storage::MatrixFree(entries) => {
                for entry in entries {
                    dense[entry.row * self.shape.1 + entry.column] = entry.value;
                }
            }
        }
        dense
    }

    pub fn nnz(&self) -> usize {
        match &self.storage {
            Storage::Dense(values) => values
                .iter()
                .filter(|value| value.norm() > f64::EPSILON)
                .count(),
            Storage::Csc { values, .. } | Storage::Csr { values, .. } => values.len(),
            Storage::Dia { values_by_row, .. } => values_by_row
                .iter()
                .filter(|value| value.norm() > f64::EPSILON)
                .count(),
            Storage::MatrixFree(entries) => entries.len(),
        }
    }

    /// Convert the operator without changing its numerical values.
    pub fn converted(&self, format: MatrixFormat) -> Result<Self> {
        if format == self.format {
            return Ok(self.clone());
        }
        let (rows, columns) = self.shape;
        let dense = self.to_dense();
        if format == MatrixFormat::Dense {
            return Self::from_dense(rows, columns, dense);
        }
        let triplets = dense.into_iter().enumerate().filter_map(|(index, value)| {
            (value.norm() > f64::EPSILON).then_some((index / columns, index % columns, value))
        });
        Self::from_triplets(rows, columns, triplets, format)
    }

    pub fn diagonal(&self) -> Vec<Complex64> {
        let dimension = self.shape.0.min(self.shape.1);
        (0..dimension)
            .map(|index| self.value_at(index, index))
            .collect()
    }

    pub fn scaled(&self, coefficient: impl Into<Complex64>) -> Result<Self> {
        let coefficient = coefficient.into();
        if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
            return Err(QuSpinError::InvalidOptions(
                "operator scale must be finite".into(),
            ));
        }
        let values = self
            .to_dense()
            .into_iter()
            .map(|value| coefficient * value)
            .collect();
        Self::from_dense(self.shape.0, self.shape.1, values)?.converted(self.format)
    }

    pub fn add(&self, right: &Self) -> Result<Self> {
        self.combine(right, Complex64::new(1.0, 0.0))
    }

    pub fn subtract(&self, right: &Self) -> Result<Self> {
        self.combine(right, Complex64::new(-1.0, 0.0))
    }

    fn combine(&self, right: &Self, right_scale: Complex64) -> Result<Self> {
        if self.shape != right.shape {
            return Err(QuSpinError::DimensionMismatch(
                "operator addition requires equal shapes".into(),
            ));
        }
        let values = self
            .to_dense()
            .into_iter()
            .zip(right.to_dense())
            .map(|(left, right)| left + right_scale * right)
            .collect();
        Self::from_dense(self.shape.0, self.shape.1, values)?.converted(self.format)
    }

    /// Matrix product `self * right`.
    pub fn product(&self, right: &Self) -> Result<Self> {
        if self.shape.1 != right.shape.0 {
            return Err(QuSpinError::DimensionMismatch(
                "operator product has incompatible inner dimensions".into(),
            ));
        }
        let left_values = self.to_dense();
        let right_values = right.to_dense();
        let mut values = vec![Complex64::new(0.0, 0.0); self.shape.0 * right.shape.1];
        for row in 0..self.shape.0 {
            for middle in 0..self.shape.1 {
                let left = left_values[row * self.shape.1 + middle];
                if left.norm() <= f64::EPSILON {
                    continue;
                }
                for column in 0..right.shape.1 {
                    values[row * right.shape.1 + column] +=
                        left * right_values[middle * right.shape.1 + column];
                }
            }
        }
        Self::from_dense(self.shape.0, right.shape.1, values)?.converted(self.format)
    }

    pub fn pow(&self, exponent: u32) -> Result<Self> {
        if self.shape.0 != self.shape.1 {
            return Err(QuSpinError::DimensionMismatch(
                "operator powers require a square operator".into(),
            ));
        }
        let dimension = self.shape.0;
        let mut identity = vec![Complex64::new(0.0, 0.0); dimension * dimension];
        for index in 0..dimension {
            identity[index * dimension + index] = Complex64::new(1.0, 0.0);
        }
        let mut result = Self::from_dense(dimension, dimension, identity)?;
        let mut base = self.clone();
        let mut remaining = exponent;
        while remaining > 0 {
            if remaining & 1 == 1 {
                result = result.product(&base)?;
            }
            remaining >>= 1;
            if remaining > 0 {
                base = base.product(&base)?;
            }
        }
        result.converted(self.format)
    }

    pub fn adjoint(&self) -> Result<Self> {
        let dense = self.to_dense();
        let mut values = vec![Complex64::new(0.0, 0.0); dense.len()];
        for row in 0..self.shape.0 {
            for column in 0..self.shape.1 {
                values[column * self.shape.0 + row] = dense[row * self.shape.1 + column].conj();
            }
        }
        Self::from_dense(self.shape.1, self.shape.0, values)?.converted(self.format)
    }

    fn value_at(&self, row: usize, column: usize) -> Complex64 {
        match &self.storage {
            Storage::Dense(values) => values[row * self.shape.1 + column],
            Storage::Csc {
                column_offsets,
                row_indices,
                values,
            } => {
                let range = column_offsets[column]..column_offsets[column + 1];
                row_indices[range.clone()]
                    .binary_search(&row)
                    .map_or(Complex64::new(0.0, 0.0), |position| {
                        values[range.start + position]
                    })
            }
            Storage::Csr {
                row_offsets,
                column_indices,
                values,
            } => {
                let range = row_offsets[row]..row_offsets[row + 1];
                column_indices[range.clone()]
                    .binary_search(&column)
                    .map_or(Complex64::new(0.0, 0.0), |position| {
                        values[range.start + position]
                    })
            }
            Storage::Dia {
                offsets,
                values_by_row,
            } => {
                let offset = row as isize - column as isize;
                offsets
                    .binary_search(&offset)
                    .map_or(Complex64::new(0.0, 0.0), |diagonal| {
                        values_by_row[diagonal * self.shape.0 + row]
                    })
            }
            Storage::MatrixFree(entries) => entries
                .binary_search_by_key(&(row, column), |entry| (entry.row, entry.column))
                .map_or(Complex64::new(0.0, 0.0), |position| entries[position].value),
        }
    }

    fn entry_is_hermitian(
        &self,
        row: usize,
        column: usize,
        value: Complex64,
        tolerance: f64,
    ) -> bool {
        (value - self.value_at(column, row).conj()).norm() <= tolerance
    }

    pub fn is_hermitian(&self, tolerance: f64) -> bool {
        if self.shape.0 != self.shape.1 {
            return false;
        }
        match &self.storage {
            Storage::Dense(values) => {
                for row in 0..self.shape.0 {
                    for column in 0..self.shape.1 {
                        if !self.entry_is_hermitian(
                            row,
                            column,
                            values[row * self.shape.1 + column],
                            tolerance,
                        ) {
                            return false;
                        }
                    }
                }
            }
            Storage::Csc {
                column_offsets,
                row_indices,
                values,
            } => {
                for column in 0..self.shape.1 {
                    for position in column_offsets[column]..column_offsets[column + 1] {
                        if !self.entry_is_hermitian(
                            row_indices[position],
                            column,
                            values[position],
                            tolerance,
                        ) {
                            return false;
                        }
                    }
                }
            }
            Storage::Csr {
                row_offsets,
                column_indices,
                values,
            } => {
                for row in 0..self.shape.0 {
                    for position in row_offsets[row]..row_offsets[row + 1] {
                        if !self.entry_is_hermitian(
                            row,
                            column_indices[position],
                            values[position],
                            tolerance,
                        ) {
                            return false;
                        }
                    }
                }
            }
            Storage::Dia {
                offsets,
                values_by_row,
            } => {
                for (diagonal, &offset) in offsets.iter().enumerate() {
                    for row in 0..self.shape.0 {
                        let value = values_by_row[diagonal * self.shape.0 + row];
                        if value.norm() <= f64::EPSILON {
                            continue;
                        }
                        let Some(column) = row.checked_add_signed(-offset) else {
                            continue;
                        };
                        if column < self.shape.1
                            && !self.entry_is_hermitian(row, column, value, tolerance)
                        {
                            return false;
                        }
                    }
                }
            }
            Storage::MatrixFree(entries) => {
                for entry in entries {
                    if !self.entry_is_hermitian(entry.row, entry.column, entry.value, tolerance) {
                        return false;
                    }
                }
            }
        }
        true
    }
}

impl LinearOperator for Operator {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }

    fn format(&self) -> MatrixFormat {
        self.format
    }

    fn apply(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        check_apply_shape(self.shape, input, output)?;
        output.fill(Complex64::new(0.0, 0.0));
        match &self.storage {
            Storage::Dense(values) => {
                for row in 0..self.shape.0 {
                    for column in 0..self.shape.1 {
                        output[row] += values[row * self.shape.1 + column] * input[column];
                    }
                }
            }
            Storage::Csc {
                column_offsets,
                row_indices,
                values,
            } => {
                for column in 0..self.shape.1 {
                    let input_value = input[column];
                    for position in column_offsets[column]..column_offsets[column + 1] {
                        output[row_indices[position]] += values[position] * input_value;
                    }
                }
            }
            Storage::Csr {
                row_offsets,
                column_indices,
                values,
            } => {
                for row in 0..self.shape.0 {
                    for position in row_offsets[row]..row_offsets[row + 1] {
                        output[row] += values[position] * input[column_indices[position]];
                    }
                }
            }
            Storage::Dia {
                offsets,
                values_by_row,
            } => {
                for (diagonal, &offset) in offsets.iter().enumerate() {
                    for row in 0..self.shape.0 {
                        let Some(column) = row.checked_add_signed(-offset) else {
                            continue;
                        };
                        if column < self.shape.1 {
                            output[row] +=
                                values_by_row[diagonal * self.shape.0 + row] * input[column];
                        }
                    }
                }
            }
            Storage::MatrixFree(entries) => {
                for entry in entries {
                    output[entry.row] += entry.value * input[entry.column];
                }
            }
        }
        Ok(())
    }

    fn shifted_solver(&self, shift: f64) -> Result<Option<Box<dyn ShiftedLinearSolver>>> {
        if self.shape.0 != self.shape.1 || !shift.is_finite() {
            return Err(QuSpinError::InvalidOptions(
                "shifted factorization requires a square operator and finite shift".into(),
            ));
        }
        let Storage::Csc {
            column_offsets,
            row_indices,
            values,
        } = &self.storage
        else {
            return Ok(None);
        };
        let dimension = self.shape.0;
        let mut triplets = Vec::with_capacity(values.len() + dimension);
        for column in 0..dimension {
            let mut has_diagonal = false;
            for position in column_offsets[column]..column_offsets[column + 1] {
                let row = row_indices[position];
                let mut value = values[position];
                if row == column {
                    value -= shift;
                    has_diagonal = true;
                }
                triplets.push(FaerTriplet::new(row, column, value));
            }
            if !has_diagonal {
                triplets.push(FaerTriplet::new(
                    column,
                    column,
                    Complex64::new(-shift, 0.0),
                ));
            }
        }
        let matrix = SparseColMat::<usize, Complex64>::try_new_from_triplets(
            dimension, dimension, &triplets,
        )
        .map_err(|error| {
            QuSpinError::UnsupportedBackend(format!(
                "could not construct sparse shifted matrix: {error}"
            ))
        })?;
        let factorization = matrix.sp_lu().map_err(|_| QuSpinError::NonConvergence {
            iterations: 0,
            residual: f64::INFINITY,
        })?;
        Ok(Some(Box::new(FaerShiftedSolver {
            factorization,
            dimension,
        })))
    }
}

fn csc_storage(entries: &[Triplet], columns: usize) -> Storage {
    let mut column_offsets = vec![0_usize; columns + 1];
    for entry in entries {
        column_offsets[entry.column + 1] += 1;
    }
    for column in 0..columns {
        column_offsets[column + 1] += column_offsets[column];
    }
    Storage::Csc {
        column_offsets,
        row_indices: entries.iter().map(|entry| entry.row).collect(),
        values: entries.iter().map(|entry| entry.value).collect(),
    }
}

fn csr_storage(entries: &[Triplet], rows: usize) -> Storage {
    let mut row_offsets = vec![0_usize; rows + 1];
    for entry in entries {
        row_offsets[entry.row + 1] += 1;
    }
    for row in 0..rows {
        row_offsets[row + 1] += row_offsets[row];
    }
    Storage::Csr {
        row_offsets,
        column_indices: entries.iter().map(|entry| entry.column).collect(),
        values: entries.iter().map(|entry| entry.value).collect(),
    }
}

fn dia_storage(entries: &[Triplet], rows: usize) -> Storage {
    let mut offsets: Vec<_> = entries
        .iter()
        .map(|entry| entry.row as isize - entry.column as isize)
        .collect();
    offsets.sort_unstable();
    offsets.dedup();
    let diagonal_index: HashMap<_, _> = offsets
        .iter()
        .copied()
        .enumerate()
        .map(|(index, offset)| (offset, index))
        .collect();
    let mut values_by_row = vec![Complex64::new(0.0, 0.0); offsets.len() * rows];
    for entry in entries {
        let offset = entry.row as isize - entry.column as isize;
        let diagonal = diagonal_index[&offset];
        values_by_row[diagonal * rows + entry.row] = entry.value;
    }
    Storage::Dia {
        offsets,
        values_by_row,
    }
}

pub(crate) fn check_apply_shape(
    shape: (usize, usize),
    input: &[Complex64],
    output: &[Complex64],
) -> Result<()> {
    if input.len() != shape.1 || output.len() != shape.0 {
        return Err(QuSpinError::DimensionMismatch(format!(
            "shape {shape:?} requires input length {} and output length {}, got {} and {}",
            shape.1,
            shape.0,
            input.len(),
            output.len()
        )));
    }
    Ok(())
}

pub(crate) fn materialize_dense(
    operator: &(impl LinearOperator + ?Sized),
) -> Result<Vec<Complex64>> {
    let (rows, columns) = operator.shape();
    let mut dense = vec![Complex64::new(0.0, 0.0); rows * columns];
    let mut input = vec![Complex64::new(0.0, 0.0); columns];
    let mut output = vec![Complex64::new(0.0, 0.0); rows];
    for column in 0..columns {
        input.fill(Complex64::new(0.0, 0.0));
        input[column] = Complex64::new(1.0, 0.0);
        operator.apply(&input, &mut output)?;
        for row in 0..rows {
            dense[row * columns + column] = output[row];
        }
    }
    Ok(dense)
}

/// Universal square or cross-sector operator builder.
pub struct OperatorBuilder<'a, Source, Target>
where
    Source: Basis,
    Target: Basis<State = Source::State>,
{
    source: &'a Source,
    target: &'a Target,
    terms: Vec<OperatorTerm>,
    checks: AssemblyChecks,
}

impl<'a, BasisType> OperatorBuilder<'a, BasisType, BasisType>
where
    BasisType: Basis,
{
    pub fn on(basis: &'a BasisType) -> Self {
        Self {
            source: basis,
            target: basis,
            terms: Vec::new(),
            checks: AssemblyChecks::all(),
        }
    }
}

impl<'a, Source, Target> OperatorBuilder<'a, Source, Target>
where
    Source: Basis,
    Target: Basis<State = Source::State>,
{
    pub fn between(source: &'a Source, target: &'a Target) -> Self {
        Self {
            source,
            target,
            terms: Vec::new(),
            checks: AssemblyChecks::rectangular(),
        }
    }

    pub fn terms(mut self, terms: impl IntoIterator<Item = OperatorTerm>) -> Self {
        self.terms.extend(terms);
        self
    }

    pub fn term(mut self, term: OperatorTerm) -> Self {
        self.terms.push(term);
        self
    }

    pub const fn checks(mut self, checks: AssemblyChecks) -> Self {
        self.checks = checks;
        self
    }

    pub fn build(self, format: MatrixFormat) -> Result<Operator> {
        let shape = (self.target.len(), self.source.len());
        let mut accumulated: HashMap<(usize, usize), Complex64> = HashMap::new();
        for column in 0..self.source.len() {
            let source_state = self.source.state(column)?;
            for term in &self.terms {
                for coupling in term.couplings() {
                    for (target_state, local_amplitude) in self.source.apply_local_transitions(
                        source_state,
                        term.operator(),
                        &coupling.sites,
                    )? {
                        let row = match self.target.index(target_state) {
                            Ok(index) => index,
                            Err(QuSpinError::StateNotInBasis) => continue,
                            Err(error) => return Err(error),
                        };
                        *accumulated
                            .entry((row, column))
                            .or_insert(Complex64::new(0.0, 0.0)) +=
                            coupling.coefficient * local_amplitude;
                    }
                }
            }
        }
        let mut entries: Vec<_> = accumulated
            .into_iter()
            .filter_map(|((row, column), value)| {
                (value.norm() > f64::EPSILON).then_some(Triplet { row, column, value })
            })
            .collect();
        match format {
            MatrixFormat::Csc => entries.sort_by_key(|entry| (entry.column, entry.row)),
            MatrixFormat::Csr => entries.sort_by_key(|entry| (entry.row, entry.column)),
            _ => entries.sort_by_key(|entry| (entry.row, entry.column)),
        }
        if format == MatrixFormat::Dia
            && entries
                .iter()
                .map(|entry| entry.row as isize - entry.column as isize)
                .collect::<std::collections::HashSet<_>>()
                .len()
                > shape.0.min(shape.1).saturating_mul(2).max(1)
        {
            return Err(QuSpinError::UnsupportedBackend(
                "the requested operator is not usefully diagonal-banded".into(),
            ));
        }
        let storage = match format {
            MatrixFormat::Dense => {
                let mut dense = vec![Complex64::new(0.0, 0.0); shape.0 * shape.1];
                for entry in &entries {
                    dense[entry.row * shape.1 + entry.column] = entry.value;
                }
                Storage::Dense(dense)
            }
            MatrixFormat::Csc => csc_storage(&entries, shape.1),
            MatrixFormat::Csr => csr_storage(&entries, shape.0),
            MatrixFormat::Dia => dia_storage(&entries, shape.0),
            MatrixFormat::MatrixFree => Storage::MatrixFree(entries),
        };
        let operator = Operator {
            shape,
            format,
            storage,
        };
        if self.checks.hermiticity && !operator.is_hermitian(1.0e-12) {
            return Err(QuSpinError::NonHermitian);
        }
        Ok(operator)
    }

    /// Assemble static and driven terms through the same local-action path.
    pub fn build_dynamic(
        self,
        dynamic_terms: impl IntoIterator<Item = DynamicTerm>,
        format: MatrixFormat,
    ) -> Result<Hamiltonian<Dynamic>> {
        let Self {
            source,
            target,
            terms,
            checks,
        } = self;
        if source.len() != target.len() {
            return Err(QuSpinError::DimensionMismatch(
                "a Hamiltonian must be square".into(),
            ));
        }
        let static_part = Self {
            source,
            target,
            terms,
            checks,
        }
        .build(format)?;
        let component_checks = AssemblyChecks {
            hermiticity: false,
            particle_conservation: checks.particle_conservation,
            symmetry_compatibility: checks.symmetry_compatibility,
        };
        let mut components = Vec::new();
        for dynamic_term in dynamic_terms {
            let (term, drive) = dynamic_term.into_parts();
            let operator = Self {
                source,
                target,
                terms: vec![term],
                checks: component_checks,
            }
            .build(format)?;
            components.push(DynamicComponent { operator, drive });
        }
        Hamiltonian::<Dynamic>::new(static_part, components)
    }
}

type DriveFunction = Arc<dyn Fn(f64) -> Complex64 + Send + Sync>;

/// A parsed local term multiplied by a scalar function of time.
#[derive(Clone)]
pub struct DynamicTerm {
    term: OperatorTerm,
    drive: DriveFunction,
}

impl std::fmt::Debug for DynamicTerm {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DynamicTerm")
            .field("term", &self.term)
            .field("drive", &"<callable>")
            .finish()
    }
}

impl DynamicTerm {
    pub fn new<F>(term: OperatorTerm, drive: F) -> Self
    where
        F: Fn(f64) -> Complex64 + Send + Sync + 'static,
    {
        Self {
            term,
            drive: Arc::new(drive),
        }
    }

    pub fn coefficient_at(&self, time: f64) -> Result<Complex64> {
        finite_drive_value(time, (self.drive)(time))
    }

    fn into_parts(self) -> (OperatorTerm, DriveFunction) {
        (self.term, self.drive)
    }
}

#[derive(Clone)]
pub struct DynamicComponent {
    operator: Operator,
    drive: DriveFunction,
}

impl std::fmt::Debug for DynamicComponent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DynamicComponent")
            .field("operator", &self.operator)
            .field("drive", &"<callable>")
            .finish()
    }
}

impl DynamicComponent {
    pub fn new<F>(operator: Operator, drive: F) -> Self
    where
        F: Fn(f64) -> Complex64 + Send + Sync + 'static,
    {
        Self {
            operator,
            drive: Arc::new(drive),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Static;

#[derive(Clone, Copy, Debug)]
pub struct Dynamic;

/// Static or explicitly time-dependent Hamiltonian with a type-state marker.
#[derive(Clone, Debug)]
pub struct Hamiltonian<Kind = Static> {
    static_part: Operator,
    dynamic_parts: Vec<DynamicComponent>,
    marker: PhantomData<Kind>,
}

impl Hamiltonian<Static> {
    pub fn new(operator: Operator) -> Result<Self> {
        let shape = operator.shape();
        if shape.0 != shape.1 {
            return Err(QuSpinError::DimensionMismatch(
                "a Hamiltonian must be square".into(),
            ));
        }
        Ok(Self {
            static_part: operator,
            dynamic_parts: Vec::new(),
            marker: PhantomData,
        })
    }

    pub fn operator(&self) -> &Operator {
        &self.static_part
    }
}

impl Hamiltonian<Dynamic> {
    pub fn new(static_part: Operator, dynamic_parts: Vec<DynamicComponent>) -> Result<Self> {
        let shape = static_part.shape();
        if shape.0 != shape.1
            || dynamic_parts
                .iter()
                .any(|component| component.operator.shape() != shape)
        {
            return Err(QuSpinError::DimensionMismatch(
                "all Hamiltonian components must share one square shape".into(),
            ));
        }
        Ok(Self {
            static_part,
            dynamic_parts,
            marker: PhantomData,
        })
    }

    pub fn evaluate(&self, time: f64, format: MatrixFormat) -> Result<Operator> {
        if !time.is_finite() {
            return Err(QuSpinError::InvalidOptions(
                "evaluation time must be finite".into(),
            ));
        }
        let shape = self.static_part.shape();
        let mut values = self.static_part.to_dense();
        for component in &self.dynamic_parts {
            let coefficient = finite_drive_value(time, (component.drive)(time))?;
            for (value, driven) in values.iter_mut().zip(component.operator.to_dense()) {
                *value += coefficient * driven;
            }
        }
        Operator::from_dense(shape.0, shape.1, values)?.converted(format)
    }

    pub fn dynamic_components(&self) -> usize {
        self.dynamic_parts.len()
    }
}

fn finite_drive_value(time: f64, value: Complex64) -> Result<Complex64> {
    if !time.is_finite() || !value.re.is_finite() || !value.im.is_finite() {
        return Err(QuSpinError::InvalidOptions(
            "drive time and coefficient must be finite".into(),
        ));
    }
    Ok(value)
}

impl LinearOperator for Hamiltonian<Static> {
    fn shape(&self) -> (usize, usize) {
        self.static_part.shape()
    }

    fn format(&self) -> MatrixFormat {
        self.static_part.format()
    }

    fn apply(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        self.static_part.apply(input, output)
    }

    fn shifted_solver(&self, shift: f64) -> Result<Option<Box<dyn ShiftedLinearSolver>>> {
        self.static_part.shifted_solver(shift)
    }
}

impl TimeDependentOperator for Hamiltonian<Static> {
    fn shape(&self) -> (usize, usize) {
        self.static_part.shape()
    }

    fn apply_at(&self, time: f64, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        if !time.is_finite() {
            return Err(QuSpinError::InvalidOptions(
                "evaluation time must be finite".into(),
            ));
        }
        self.static_part.apply(input, output)
    }
}

impl TimeDependentOperator for Hamiltonian<Dynamic> {
    fn shape(&self) -> (usize, usize) {
        self.static_part.shape()
    }

    fn apply_at(&self, time: f64, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        check_apply_shape(self.static_part.shape(), input, output)?;
        self.static_part.apply(input, output)?;
        let mut driven = vec![Complex64::new(0.0, 0.0); output.len()];
        for component in &self.dynamic_parts {
            let coefficient = finite_drive_value(time, (component.drive)(time))?;
            component.operator.apply(input, &mut driven)?;
            for (value, contribution) in output.iter_mut().zip(&driven) {
                *value += coefficient * *contribution;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct QuantumComponent {
    name: String,
    operator: Operator,
    default: Option<Complex64>,
}

impl QuantumComponent {
    pub fn required(name: impl Into<String>, operator: Operator) -> Self {
        Self {
            name: name.into(),
            operator,
            default: None,
        }
    }

    pub fn with_default(
        name: impl Into<String>,
        operator: Operator,
        default: impl Into<Complex64>,
    ) -> Self {
        Self {
            name: name.into(),
            operator,
            default: Some(default.into()),
        }
    }
}

/// Named linear combination of operator components.
#[derive(Clone, Debug)]
pub struct QuantumOperator {
    components: Vec<QuantumComponent>,
    shape: (usize, usize),
}

impl QuantumOperator {
    pub fn new(components: impl IntoIterator<Item = QuantumComponent>) -> Result<Self> {
        let components: Vec<_> = components.into_iter().collect();
        let first = components.first().ok_or_else(|| {
            QuSpinError::InvalidOptions("QuantumOperator requires at least one component".into())
        })?;
        let shape = first.operator.shape();
        let mut names = std::collections::HashSet::new();
        for component in &components {
            if component.name.is_empty() || !names.insert(component.name.clone()) {
                return Err(QuSpinError::InvalidOptions(
                    "component names must be nonempty and unique".into(),
                ));
            }
            if component.operator.shape() != shape {
                return Err(QuSpinError::DimensionMismatch(
                    "all parameterized components must have equal shapes".into(),
                ));
            }
            if component
                .default
                .is_some_and(|value| !value.re.is_finite() || !value.im.is_finite())
            {
                return Err(QuSpinError::InvalidOptions(
                    "component defaults must be finite".into(),
                ));
            }
        }
        Ok(Self { components, shape })
    }

    pub const fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn evaluate(
        &self,
        parameters: &HashMap<String, Complex64>,
        format: MatrixFormat,
    ) -> Result<Operator> {
        if let Some(name) = parameters.keys().find(|name| {
            !self
                .components
                .iter()
                .any(|component| &component.name == *name)
        }) {
            return Err(QuSpinError::InvalidOptions(format!(
                "unknown operator parameter {name:?}"
            )));
        }
        let mut values = vec![Complex64::new(0.0, 0.0); self.shape.0 * self.shape.1];
        for component in &self.components {
            let coefficient = parameters
                .get(&component.name)
                .copied()
                .or(component.default)
                .ok_or_else(|| {
                    QuSpinError::InvalidOptions(format!(
                        "missing required operator parameter {:?}",
                        component.name
                    ))
                })?;
            if !coefficient.re.is_finite() || !coefficient.im.is_finite() {
                return Err(QuSpinError::InvalidOptions(format!(
                    "operator parameter {:?} must be finite",
                    component.name
                )));
            }
            for (value, basis_value) in values.iter_mut().zip(component.operator.to_dense()) {
                *value += coefficient * basis_value;
            }
        }
        Operator::from_dense(self.shape.0, self.shape.1, values)?.converted(format)
    }
}

pub fn commutator(left: &Operator, right: &Operator) -> Result<Operator> {
    left.product(right)?.subtract(&right.product(left)?)
}

pub fn anticommutator(left: &Operator, right: &Operator) -> Result<Operator> {
    left.product(right)?.add(&right.product(left)?)
}
