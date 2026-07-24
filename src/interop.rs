//! Runtime-owned exact-diagonalization model shared by language frontends.
//!
//! The native generic API remains the zero-cost path. This module provides a
//! small owned narrow waist for frontends that select a packed basis at
//! runtime and need to reuse one mathematical model across materialization and
//! solver operations.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::basis::{Basis, PackedBasis};
use crate::operator::{AssemblyChecks, MatrixFormat, Operator, OperatorBuilder, OperatorSpec};
use crate::solve::{Eigensystem, EighOptions, EigshOptions, eigh_with_options, eigsh};
use crate::{QmbedError, Result};

/// One owned basis and operator specification reusable across frontend calls.
#[derive(Clone, Debug)]
pub struct PackedEdModel {
    basis: PackedBasis,
    terms: Vec<OperatorSpec>,
    checks: AssemblyChecks,
    operators: Arc<Mutex<HashMap<MatrixFormat, Arc<Operator>>>>,
}

impl PackedEdModel {
    pub fn new(
        basis: impl Into<PackedBasis>,
        terms: impl IntoIterator<Item = OperatorSpec>,
    ) -> Self {
        Self {
            basis: basis.into(),
            terms: terms.into_iter().collect(),
            checks: AssemblyChecks::all(),
            operators: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn with_checks(mut self, checks: AssemblyChecks) -> Self {
        self.checks = checks;
        self.operators = Arc::new(Mutex::new(HashMap::new()));
        self
    }

    pub fn with_site_permutation(mut self, permutation: &[usize]) -> Result<Self> {
        self.terms = self
            .terms
            .iter()
            .map(|term| term.with_site_permutation(permutation))
            .collect::<Result<Vec<_>>>()?;
        self.operators = Arc::new(Mutex::new(HashMap::new()));
        Ok(self)
    }

    pub const fn basis(&self) -> &PackedBasis {
        &self.basis
    }

    pub fn terms(&self) -> &[OperatorSpec] {
        &self.terms
    }

    pub fn dimension(&self) -> usize {
        self.basis.len()
    }

    pub fn states(&self) -> Result<Vec<u128>> {
        (0..self.basis.len())
            .map(|index| self.basis.state(index))
            .collect()
    }

    /// Return the assembled operator shared by all calls using this model.
    ///
    /// Assembly is performed at most once per storage format. Clones of an
    /// unchanged model share the same cache; model-transforming builders reset
    /// it before changing checks or site labels.
    pub fn materialized(&self, format: MatrixFormat) -> Result<Arc<Operator>> {
        let mut operators = self.operators.lock().map_err(|_| {
            QmbedError::InternalState("materialized-operator cache lock is poisoned".into())
        })?;
        if let Some(operator) = operators.get(&format) {
            return Ok(Arc::clone(operator));
        }
        let operator = Arc::new(
            OperatorBuilder::on(&self.basis)
                .terms(self.terms.clone())
                .checks(self.checks)
                .build(format)?,
        );
        operators.insert(format, Arc::clone(&operator));
        Ok(operator)
    }

    /// Materialize an owned operator for callers that do not need reuse.
    pub fn materialize(&self, format: MatrixFormat) -> Result<Operator> {
        Ok((*self.materialized(format)?).clone())
    }

    pub fn eigh(&self, options: EighOptions) -> Result<Eigensystem> {
        let operator = self.materialized(MatrixFormat::Dense)?;
        eigh_with_options(operator.as_ref(), options)
    }

    pub fn eigsh(&self, format: MatrixFormat, options: EigshOptions) -> Result<Eigensystem> {
        let operator = self.materialized(format)?;
        eigsh(operator.as_ref(), options)
    }
}
