use std::f64::consts::PI;
use std::sync::Arc;

use num_complex::Complex64;

use crate::operator::{LinearOperator, MatrixFormat};
use crate::solve::{
    EvolutionOptions, expm_action, lanczos_spectral_measure, real_symmetric_eigenpairs_all,
};
use crate::{QuSpinError, Result};

pub struct DriveStep {
    pub hamiltonian: Arc<dyn LinearOperator>,
    pub duration: f64,
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
    steps: Vec<DriveStep>,
    dimension: usize,
    evolution: EvolutionOptions,
}

impl Floquet {
    pub fn new(steps: impl IntoIterator<Item = DriveStep>) -> Result<Self> {
        let steps: Vec<_> = steps.into_iter().collect();
        let first = steps.first().ok_or_else(|| {
            QuSpinError::InvalidOptions("Floquet requires at least one drive step".into())
        })?;
        let dimension = first.hamiltonian.shape().0;
        if steps
            .iter()
            .any(|step| step.hamiltonian.shape() != (dimension, dimension))
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
            state = expm_action(
                step.hamiltonian.as_ref(),
                &state,
                step.duration,
                &self.evolution,
            )?;
        }
        output.copy_from_slice(&state);
        Ok(())
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
        let (energies, eigenvectors) = real_symmetric_eigenpairs_all(target_hamiltonian)?;
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
