use std::f64::consts::PI;
use std::sync::Arc;

use num_complex::Complex64;

use crate::backend;
use crate::operator::{
    LinearOperator, MatrixFormat, Operator, TimeDependentOperator, materialize_dense,
};
use crate::solve::{
    EvolutionOptions, evolve, evolve_time_dependent, expm_action, hermitian_eigenpairs_all,
    lanczos_spectral_measure,
};
use crate::{QuSpinError, Result};

const DENSE_PROPAGATOR_CUTOFF: usize = 128;

pub struct DriveStep {
    pub hamiltonian: Arc<dyn LinearOperator>,
    pub duration: f64,
}

pub struct CallableDriveStep {
    pub hamiltonian: Arc<dyn TimeDependentOperator>,
    pub duration: f64,
}

impl CallableDriveStep {
    pub fn new(hamiltonian: Arc<dyn TimeDependentOperator>, duration: f64) -> Result<Self> {
        let shape = hamiltonian.shape();
        if shape.0 != shape.1 {
            return Err(QuSpinError::DimensionMismatch(
                "a callable drive Hamiltonian must be square".into(),
            ));
        }
        if !duration.is_finite() || duration < 0.0 {
            return Err(QuSpinError::InvalidOptions(
                "drive duration must be finite and nonnegative".into(),
            ));
        }
        Ok(Self {
            hamiltonian,
            duration,
        })
    }
}

enum FloquetStep {
    Static(DriveStep),
    Callable(CallableDriveStep),
}

impl DriveStep {
    pub fn new(hamiltonian: Arc<dyn LinearOperator>, duration: f64) -> Result<Self> {
        let shape = hamiltonian.shape();
        if shape.0 != shape.1 {
            return Err(QuSpinError::DimensionMismatch(
                "a drive Hamiltonian must be square".into(),
            ));
        }
        if !duration.is_finite() || duration < 0.0 {
            return Err(QuSpinError::InvalidOptions(
                "drive duration must be finite and nonnegative".into(),
            ));
        }
        Ok(Self {
            hamiltonian,
            duration,
        })
    }
}

/// One period of a piecewise-constant drive.
pub struct Floquet {
    steps: Vec<FloquetStep>,
    dimension: usize,
    evolution: EvolutionOptions,
}

impl Floquet {
    pub fn new(steps: impl IntoIterator<Item = DriveStep>) -> Result<Self> {
        let steps: Vec<_> = steps.into_iter().map(FloquetStep::Static).collect();
        let first = steps.first().ok_or_else(|| {
            QuSpinError::InvalidOptions("Floquet requires at least one drive step".into())
        })?;
        let dimension = first.shape().0;
        if steps
            .iter()
            .any(|step| step.shape() != (dimension, dimension))
        {
            return Err(QuSpinError::DimensionMismatch(
                "all drive steps must have the same square shape".into(),
            ));
        }
        Ok(Self {
            steps,
            dimension,
            evolution: EvolutionOptions {
                times: vec![0.0],
                krylov_dimension: 64,
                tolerance: 1.0e-12,
                max_substeps: 10_000,
                hamiltonian: true,
            },
        })
    }

    pub fn from_callable(steps: impl IntoIterator<Item = CallableDriveStep>) -> Result<Self> {
        let steps: Vec<_> = steps.into_iter().map(FloquetStep::Callable).collect();
        let first = steps.first().ok_or_else(|| {
            QuSpinError::InvalidOptions("Floquet requires at least one drive step".into())
        })?;
        let dimension = first.shape().0;
        if steps
            .iter()
            .any(|step| step.shape() != (dimension, dimension))
        {
            return Err(QuSpinError::DimensionMismatch(
                "all drive steps must have the same square shape".into(),
            ));
        }
        Ok(Self {
            steps,
            dimension,
            evolution: EvolutionOptions {
                times: vec![0.0],
                krylov_dimension: 64,
                tolerance: 1.0e-12,
                max_substeps: 10_000,
                hamiltonian: true,
            },
        })
    }

    pub fn with_evolution_options(mut self, options: EvolutionOptions) -> Self {
        self.evolution = options;
        self.evolution.hamiltonian = true;
        self
    }

    pub fn apply_period(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        if input.len() != self.dimension || output.len() != self.dimension {
            return Err(QuSpinError::DimensionMismatch(
                "Floquet input or output length does not match".into(),
            ));
        }
        let mut state = input.to_vec();
        for step in &self.steps {
            match step {
                FloquetStep::Static(step) => {
                    state = expm_action(
                        step.hamiltonian.as_ref(),
                        &state,
                        step.duration,
                        &self.evolution,
                    )?;
                }
                FloquetStep::Callable(step) => {
                    let mut options = self.evolution.clone();
                    options.times = vec![step.duration];
                    state = evolve_time_dependent(step.hamiltonian.as_ref(), &state, options)?
                        .states
                        .pop()
                        .ok_or(QuSpinError::NonConvergence {
                            iterations: 0,
                            residual: f64::INFINITY,
                        })?;
                }
            }
        }
        output.copy_from_slice(&state);
        Ok(())
    }

    pub fn period(&self) -> f64 {
        self.steps.iter().map(FloquetStep::duration).sum()
    }

    pub fn full_unitary(&self, format: MatrixFormat) -> Result<Operator> {
        let dense = if self
            .steps
            .iter()
            .all(|step| matches!(step, FloquetStep::Static(_)))
            && self.dimension <= DENSE_PROPAGATOR_CUTOFF
        {
            let mut total = vec![Complex64::new(0.0, 0.0); self.dimension * self.dimension];
            for index in 0..self.dimension {
                total[index * self.dimension + index] = Complex64::new(1.0, 0.0);
            }
            for step in &self.steps {
                let FloquetStep::Static(step) = step else {
                    unreachable!("all Floquet steps were checked as static");
                };
                let hamiltonian = materialize_dense(step.hamiltonian.as_ref())?;
                let propagator = backend::hermitian_exponential(
                    &hamiltonian,
                    self.dimension,
                    Complex64::new(0.0, -step.duration),
                )?;
                total = backend::square_matmul(&propagator, &total, self.dimension)?;
            }
            total
        } else {
            materialize_dense(self)?
        };
        Operator::from_dense(self.dimension, self.dimension, dense)?.converted(format)
    }

    pub fn eigensystem(&self) -> Result<FloquetEigensystem> {
        let period = self.period();
        if period <= 0.0 {
            return Err(QuSpinError::InvalidOptions(
                "Floquet eigensystems require a positive period".into(),
            ));
        }
        let unitary = materialize_dense(&self.full_unitary(MatrixFormat::Dense)?)?;
        let eigensystem = backend::complex_eigenpairs(&unitary, self.dimension)?;
        let mut entries = Vec::with_capacity(self.dimension);
        for column in 0..self.dimension {
            let eigenvalue = eigensystem.eigenvalues[column];
            if (eigenvalue.norm() - 1.0).abs() > 1.0e-8 {
                return Err(QuSpinError::NonConvergence {
                    iterations: 1,
                    residual: (eigenvalue.norm() - 1.0).abs(),
                });
            }
            let vector = eigensystem.eigenvectors[column].clone();
            let mut applied = vec![Complex64::new(0.0, 0.0); self.dimension];
            for row in 0..self.dimension {
                applied[row] = (0..self.dimension)
                    .map(|inner| unitary[row * self.dimension + inner] * vector[inner])
                    .sum();
            }
            let residual = applied
                .iter()
                .zip(&vector)
                .map(|(actual, component)| (*actual - eigenvalue * *component).norm_sqr())
                .sum::<f64>()
                .sqrt();
            entries.push((-eigenvalue.arg() / period, eigenvalue, vector, residual));
        }
        entries.sort_by(|left, right| left.0.total_cmp(&right.0));
        Ok(FloquetEigensystem {
            quasienergies: entries.iter().map(|entry| entry.0).collect(),
            eigenvalues: entries.iter().map(|entry| entry.1).collect(),
            eigenvectors: entries.iter().map(|entry| entry.2.clone()).collect(),
            residuals: entries.into_iter().map(|entry| entry.3).collect(),
        })
    }

    pub fn effective_hamiltonian(&self, format: MatrixFormat) -> Result<Operator> {
        let eigensystem = self.eigensystem()?;
        let mut values = vec![Complex64::new(0.0, 0.0); self.dimension * self.dimension];
        for (energy, vector) in eigensystem
            .quasienergies
            .iter()
            .zip(&eigensystem.eigenvectors)
        {
            for row in 0..self.dimension {
                for column in 0..self.dimension {
                    values[row * self.dimension + column] +=
                        *energy * vector[row] * vector[column].conj();
                }
            }
        }
        Operator::from_dense(self.dimension, self.dimension, values)?.converted(format)
    }
}

impl FloquetStep {
    fn shape(&self) -> (usize, usize) {
        match self {
            Self::Static(step) => step.hamiltonian.shape(),
            Self::Callable(step) => step.hamiltonian.shape(),
        }
    }

    fn duration(&self) -> f64 {
        match self {
            Self::Static(step) => step.duration,
            Self::Callable(step) => step.duration,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FloquetEigensystem {
    pub quasienergies: Vec<f64>,
    pub eigenvalues: Vec<Complex64>,
    pub eigenvectors: Vec<Vec<Complex64>>,
    pub residuals: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FloquetCoordinate {
    pub cycle: usize,
    pub within_cycle: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FloquetTimeVector {
    period: f64,
    cycles: usize,
    points_per_cycle: usize,
    times: Vec<f64>,
}

impl FloquetTimeVector {
    pub fn new(
        period: f64,
        cycles: usize,
        points_per_cycle: usize,
        include_endpoint: bool,
    ) -> Result<Self> {
        if !period.is_finite() || period <= 0.0 || cycles == 0 || points_per_cycle == 0 {
            return Err(QuSpinError::InvalidOptions(
                "Floquet time-vector controls must be positive".into(),
            ));
        }
        let points = cycles
            .checked_mul(points_per_cycle)
            .and_then(|value| value.checked_add(usize::from(include_endpoint)))
            .ok_or_else(|| QuSpinError::InvalidOptions("Floquet time-vector overflow".into()))?;
        let step = period / points_per_cycle as f64;
        let times = (0..points).map(|index| index as f64 * step).collect();
        Ok(Self {
            period,
            cycles,
            points_per_cycle,
            times,
        })
    }

    pub const fn period(&self) -> f64 {
        self.period
    }

    pub const fn cycles(&self) -> usize {
        self.cycles
    }

    pub fn times(&self) -> &[f64] {
        &self.times
    }

    pub fn coordinate(&self, index: usize) -> Result<FloquetCoordinate> {
        if index >= self.times.len() {
            return Err(QuSpinError::InvalidOptions(
                "Floquet time index is out of bounds".into(),
            ));
        }
        Ok(FloquetCoordinate {
            cycle: index / self.points_per_cycle,
            within_cycle: (index % self.points_per_cycle) as f64 * self.period
                / self.points_per_cycle as f64,
        })
    }
}

impl LinearOperator for Floquet {
    fn shape(&self) -> (usize, usize) {
        (self.dimension, self.dimension)
    }

    fn format(&self) -> MatrixFormat {
        MatrixFormat::MatrixFree
    }

    fn apply(&self, input: &[Complex64], output: &mut [Complex64]) -> Result<()> {
        self.apply_period(input, output)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpectrumOptions {
    pub frequencies: Vec<f64>,
    pub reference_energy: f64,
    pub broadening: f64,
    pub krylov_dimension: usize,
    pub tolerance: f64,
}

impl SpectrumOptions {
    fn validate(&self) -> Result<()> {
        if self.frequencies.is_empty()
            || self.frequencies.iter().any(|value| !value.is_finite())
            || !self.reference_energy.is_finite()
            || !self.broadening.is_finite()
            || self.broadening <= 0.0
            || self.krylov_dimension == 0
            || !self.tolerance.is_finite()
            || self.tolerance <= 0.0
        {
            return Err(QuSpinError::InvalidOptions(
                "invalid spectrum frequency grid or numerical controls".into(),
            ));
        }
        Ok(())
    }
}

/// Lorentzian-broadened spectral density in a same or different target sector.
pub fn spectral_function<H, P>(
    target_hamiltonian: &H,
    source: &[Complex64],
    probe: &P,
    options: SpectrumOptions,
) -> Result<Vec<f64>>
where
    H: LinearOperator + ?Sized,
    P: LinearOperator + ?Sized,
{
    options.validate()?;
    let target_shape = target_hamiltonian.shape();
    let probe_shape = probe.shape();
    if target_shape.0 != target_shape.1
        || probe_shape.0 != target_shape.0
        || probe_shape.1 != source.len()
    {
        return Err(QuSpinError::DimensionMismatch(
            "target Hamiltonian, source, and probe shapes are incompatible".into(),
        ));
    }
    let mut created = vec![Complex64::new(0.0, 0.0); probe_shape.0];
    probe.apply(source, &mut created)?;
    let (energies, weights) = if target_shape.0 <= 128 {
        let (energies, eigenvectors) = hermitian_eigenpairs_all(target_hamiltonian)?;
        let weights = eigenvectors
            .iter()
            .map(|vector| {
                vector
                    .iter()
                    .zip(&created)
                    .map(|(left, right)| left.conj() * *right)
                    .sum::<Complex64>()
                    .norm_sqr()
            })
            .collect();
        (energies, weights)
    } else {
        lanczos_spectral_measure(target_hamiltonian, &created, options.krylov_dimension)?
    };
    Ok(options
        .frequencies
        .iter()
        .map(|&frequency| {
            energies
                .iter()
                .zip(&weights)
                .map(|(&energy, &weight)| {
                    let detuning = frequency + options.reference_energy - energy;
                    weight * options.broadening
                        / (PI * (detuning * detuning + options.broadening.powi(2)))
                })
                .sum()
        })
        .collect())
}

/// Real-time two-point function `<psi|A(t) B(0)|psi>`.
pub fn dynamical_correlator<H, A, B>(
    hamiltonian: &H,
    state: &[Complex64],
    left_probe: &A,
    right_probe: &B,
    mut options: EvolutionOptions,
) -> Result<Vec<Complex64>>
where
    H: LinearOperator + ?Sized,
    A: LinearOperator + ?Sized,
    B: LinearOperator + ?Sized,
{
    let dimension = state.len();
    if hamiltonian.shape() != (dimension, dimension)
        || left_probe.shape() != (dimension, dimension)
        || right_probe.shape() != (dimension, dimension)
    {
        return Err(QuSpinError::DimensionMismatch(
            "correlator Hamiltonian, probes, and state dimensions do not match".into(),
        ));
    }
    options.hamiltonian = true;
    let mut created = vec![Complex64::new(0.0, 0.0); dimension];
    right_probe.apply(state, &mut created)?;
    let reference = evolve(hamiltonian, state, options.clone())?;
    let excited = evolve(hamiltonian, &created, options)?;
    reference
        .states
        .iter()
        .zip(excited.states)
        .map(|(reference_state, excited_state)| {
            let mut probed = vec![Complex64::new(0.0, 0.0); dimension];
            left_probe.apply(&excited_state, &mut probed)?;
            Ok(reference_state
                .iter()
                .zip(probed)
                .map(|(left, right)| left.conj() * right)
                .sum())
        })
        .collect()
}
