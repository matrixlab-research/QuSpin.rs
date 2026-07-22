use std::collections::HashMap;

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
    Sparse(Vec<Triplet>),
}

/// Concrete operator returned by universal assembly.
#[derive(Clone, Debug)]
pub struct Operator {
    shape: (usize, usize),
    format: MatrixFormat,
    storage: Storage,
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

    pub fn to_dense(&self) -> Vec<Complex64> {
        match &self.storage {
            Storage::Dense(values) => values.clone(),
            Storage::Sparse(entries) => {
                let mut values = vec![Complex64::new(0.0, 0.0); self.shape.0 * self.shape.1];
                for entry in entries {
                    values[entry.row * self.shape.1 + entry.column] += entry.value;
                }
                values
            }
        }
    }

    pub fn is_hermitian(&self, tolerance: f64) -> bool {
        if self.shape.0 != self.shape.1 {
            return false;
        }
        let dense = self.to_dense();
        let dimension = self.shape.0;
        for row in 0..dimension {
            for column in 0..dimension {
                let difference =
                    dense[row * dimension + column] - dense[column * dimension + row].conj();
                if difference.norm() > tolerance {
                    return false;
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
            Storage::Sparse(entries) => {
                for entry in entries {
                    output[entry.row] += entry.value * input[entry.column];
                }
            }
        }
        Ok(())
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
                    let Some((target_state, local_amplitude)) =
                        self.source
                            .apply_local(source_state, term.operator(), &coupling.sites)?
                    else {
                        continue;
                    };
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
        let storage = if format == MatrixFormat::Dense {
            let mut dense = vec![Complex64::new(0.0, 0.0); shape.0 * shape.1];
            for entry in &entries {
                dense[entry.row * shape.1 + entry.column] = entry.value;
            }
            Storage::Dense(dense)
        } else {
            Storage::Sparse(entries)
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
}
