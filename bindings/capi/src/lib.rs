//! Stable, language-neutral entry point shared by the Python and Julia layers.

use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};

use qmbed::basis::{
    Basis, BosonBasis1D, SpinBasis1D, SpinfulFermionBasis1D, SpinlessFermionBasis1D,
};
use qmbed::operator::{
    Coupling, LocalOperator, MatrixFormat, OpProduct, OperatorBuilder, OperatorSpec,
};
use qmbed::solve::{EigshOptions, SpectrumTarget, eigsh};
use qmbed::{Complex64, QmbedError, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct SolveRequest {
    basis: BasisRequest,
    terms: Vec<TermRequest>,
    #[serde(default)]
    format: StorageFormat,
    solver: SolverRequest,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum BasisRequest {
    Spin {
        sites: usize,
        #[serde(default = "default_spin_twice")]
        spin_twice: u16,
        up: Option<usize>,
        momentum: Option<i32>,
        parity: Option<i8>,
        #[serde(default)]
        pauli: bool,
    },
    Boson {
        sites: usize,
        particles: Option<usize>,
        states_per_site: usize,
    },
    SpinlessFermion {
        sites: usize,
        particles: Option<usize>,
        momentum: Option<i32>,
    },
    SpinfulFermion {
        sites: usize,
        particles_up: Option<usize>,
        particles_down: Option<usize>,
    },
}

const fn default_spin_twice() -> u16 {
    1
}

#[derive(Debug, Deserialize)]
struct TermRequest {
    product: ProductRequest,
    couplings: Vec<CouplingRequest>,
}

#[derive(Debug, Deserialize)]
struct ProductRequest {
    local: Vec<String>,
    split: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct CouplingRequest {
    coefficient: [f64; 2],
    sites: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum StorageFormat {
    Dense,
    #[default]
    Csc,
    Csr,
    Dia,
    MatrixFree,
}

impl From<StorageFormat> for MatrixFormat {
    fn from(value: StorageFormat) -> Self {
        match value {
            StorageFormat::Dense => Self::Dense,
            StorageFormat::Csc => Self::Csc,
            StorageFormat::Csr => Self::Csr,
            StorageFormat::Dia => Self::Dia,
            StorageFormat::MatrixFree => Self::MatrixFree,
        }
    }
}

#[derive(Debug, Deserialize)]
struct SolverRequest {
    eigenpairs: usize,
    target: TargetRequest,
    krylov_dimension: Option<usize>,
    #[serde(default = "default_tolerance")]
    tolerance: f64,
    #[serde(default = "default_iterations")]
    max_iterations: usize,
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    eigenvectors: bool,
}

const fn default_tolerance() -> f64 {
    1.0e-10
}

const fn default_iterations() -> usize {
    1_000
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum TargetRequest {
    SmallestAlgebraic,
    LargestAlgebraic,
    SmallestMagnitude,
    LargestMagnitude,
    BothEnds,
    Shift { value: f64 },
}

impl From<TargetRequest> for SpectrumTarget {
    fn from(value: TargetRequest) -> Self {
        match value {
            TargetRequest::SmallestAlgebraic => Self::SmallestAlgebraic,
            TargetRequest::LargestAlgebraic => Self::LargestAlgebraic,
            TargetRequest::SmallestMagnitude => Self::SmallestMagnitude,
            TargetRequest::LargestMagnitude => Self::LargestMagnitude,
            TargetRequest::BothEnds => Self::BothEnds,
            TargetRequest::Shift { value } => Self::Shift(value),
        }
    }
}

#[derive(Debug, Serialize)]
struct SolveResult {
    dimension: usize,
    eigenvalues: Vec<f64>,
    residuals: Vec<f64>,
    iterations: usize,
    converged: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    eigenvectors: Option<Vec<Vec<[f64; 2]>>>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum Response {
    Ok { result: SolveResult },
    Error { error: String },
}

pub fn run_json(request: &str) -> String {
    let response = serde_json::from_str::<SolveRequest>(request)
        .map_err(|error| QmbedError::InvalidOptions(format!("invalid binding request: {error}")))
        .and_then(execute)
        .map_or_else(
            |error| Response::Error {
                error: error.to_string(),
            },
            |result| Response::Ok { result },
        );
    serde_json::to_string(&response).unwrap_or_else(|error| {
        format!(r#"{{"status":"error","error":"response serialization failed: {error}"}}"#)
    })
}

fn execute(request: SolveRequest) -> Result<SolveResult> {
    match request.basis {
        BasisRequest::Spin {
            sites,
            spin_twice,
            up,
            momentum,
            parity,
            pauli,
        } => {
            let mut builder = SpinBasis1D::builder(sites)
                .spin_twice(spin_twice)
                .pauli(pauli);
            if let Some(up) = up {
                builder = builder.up(up);
            }
            if let Some(momentum) = momentum {
                builder = builder.momentum(momentum);
            }
            if let Some(parity) = parity {
                builder = builder.parity(parity);
            }
            solve_on(
                &builder.build()?,
                request.terms,
                request.format,
                &request.solver,
            )
        }
        BasisRequest::Boson {
            sites,
            particles,
            states_per_site,
        } => {
            let mut builder = BosonBasis1D::builder(sites, states_per_site);
            if let Some(particles) = particles {
                builder = builder.particles(particles);
            }
            solve_on(
                &builder.build()?,
                request.terms,
                request.format,
                &request.solver,
            )
        }
        BasisRequest::SpinlessFermion {
            sites,
            particles,
            momentum,
        } => {
            let mut builder = SpinlessFermionBasis1D::builder(sites);
            if let Some(particles) = particles {
                builder = builder.particles(particles);
            }
            if let Some(momentum) = momentum {
                builder = builder.momentum(momentum);
            }
            solve_on(
                &builder.build()?,
                request.terms,
                request.format,
                &request.solver,
            )
        }
        BasisRequest::SpinfulFermion {
            sites,
            particles_up,
            particles_down,
        } => {
            let mut builder = SpinfulFermionBasis1D::builder(sites);
            if let Some(particles) = particles_up {
                builder = builder.particles_up(particles);
            }
            if let Some(particles) = particles_down {
                builder = builder.particles_down(particles);
            }
            solve_on(
                &builder.build()?,
                request.terms,
                request.format,
                &request.solver,
            )
        }
    }
}

fn solve_on<B>(
    basis: &B,
    terms: Vec<TermRequest>,
    format: StorageFormat,
    solver: &SolverRequest,
) -> Result<SolveResult>
where
    B: Basis<State = u128>,
{
    let terms = terms
        .into_iter()
        .map(typed_term)
        .collect::<Result<Vec<_>>>()?;
    let operator = OperatorBuilder::on(basis)
        .terms(terms)
        .build(format.into())?;
    let include_vectors = solver.eigenvectors;
    let result = eigsh(
        &operator,
        EigshOptions {
            eigenpairs: solver.eigenpairs,
            target: solver.target.into(),
            krylov_dimension: solver.krylov_dimension,
            tolerance: solver.tolerance,
            max_iterations: solver.max_iterations,
            seed: solver.seed,
        },
    )?;
    Ok(SolveResult {
        dimension: basis.len(),
        eigenvalues: result.eigenvalues,
        residuals: result.residuals,
        iterations: result.iterations,
        converged: result.converged,
        eigenvectors: include_vectors.then(|| {
            result
                .eigenvectors
                .into_iter()
                .map(|vector| {
                    vector
                        .into_iter()
                        .map(|value| [value.re, value.im])
                        .collect()
                })
                .collect()
        }),
    })
}

fn typed_term(term: TermRequest) -> Result<OperatorSpec> {
    let local = term
        .product
        .local
        .iter()
        .map(|name| typed_local_operator(name))
        .collect::<Result<Vec<_>>>()?;
    let product = OpProduct::with_split(local, term.product.split)?;
    let couplings = term.couplings.into_iter().map(|coupling| {
        Coupling::new(
            Complex64::new(coupling.coefficient[0], coupling.coefficient[1]),
            coupling.sites,
        )
    });
    OperatorSpec::from_product(product, couplings)
}

fn typed_local_operator(name: &str) -> Result<LocalOperator> {
    match name {
        "identity" => Ok(LocalOperator::Identity),
        "number" => Ok(LocalOperator::Number),
        "z" => Ok(LocalOperator::Z),
        "raising" => Ok(LocalOperator::Raising),
        "lowering" => Ok(LocalOperator::Lowering),
        "x" => Ok(LocalOperator::X),
        "y" => Ok(LocalOperator::Y),
        custom if custom.starts_with("custom:") => {
            let mut symbols = custom["custom:".len()..].chars();
            match (symbols.next(), symbols.next()) {
                (Some(symbol), None) => Ok(LocalOperator::Custom(symbol)),
                _ => Err(QmbedError::InvalidOperator(custom.into())),
            }
        }
        unknown => Err(QmbedError::InvalidOperator(unknown.into())),
    }
}

/// Execute one JSON request and return an owned UTF-8 JSON response.
///
/// # Safety
///
/// `request` must point to a valid NUL-terminated string for the duration of
/// this call. The returned pointer must be released exactly once with
/// [`qmbed_string_free`].
///
/// # Panics
///
/// This function panics only if Rust's JSON serializer violates its guarantee
/// to escape interior NUL characters.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qmbed_run_json(request: *const c_char) -> *mut c_char {
    let response = catch_unwind(AssertUnwindSafe(|| {
        if request.is_null() {
            return r#"{"status":"error","error":"request pointer is null"}"#.to_string();
        }
        // SAFETY: The caller contract requires a live NUL-terminated string.
        let request = unsafe { CStr::from_ptr(request) };
        match request.to_str() {
            Ok(request) => run_json(request),
            Err(error) => {
                format!(r#"{{"status":"error","error":"request is not UTF-8: {error}"}}"#)
            }
        }
    }))
    .unwrap_or_else(|_| r#"{"status":"error","error":"Rust binding panic"}"#.to_string());
    CString::new(response)
        .expect("serialized JSON never contains an interior NUL")
        .into_raw()
}

/// Release a response returned by [`qmbed_run_json`].
///
/// # Safety
///
/// `response` must be null or a pointer returned by [`qmbed_run_json`] that
/// has not already been released.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qmbed_string_free(response: *mut c_char) {
    if !response.is_null() {
        // SAFETY: The caller contract transfers the unique owned pointer back.
        drop(unsafe { CString::from_raw(response) });
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use super::run_json;

    #[test]
    fn typed_json_request_reaches_the_rust_solver() {
        let response = run_json(
            r#"{
                "basis":{"kind":"spin","sites":2,"pauli":false},
                "terms":[
                    {"product":{"local":["z","z"]},"couplings":[{"coefficient":[1.0,0.0],"sites":[0,1]}]},
                    {"product":{"local":["raising","lowering"]},"couplings":[{"coefficient":[0.5,0.0],"sites":[0,1]}]},
                    {"product":{"local":["lowering","raising"]},"couplings":[{"coefficient":[0.5,0.0],"sites":[0,1]}]}
                ],
                "format":"csc",
                "solver":{"eigenpairs":2,"target":{"kind":"smallest_algebraic"}}
            }"#,
        );
        let response: Value = serde_json::from_str(&response).unwrap();
        assert_eq!(response["status"], "ok");
        assert_eq!(response["result"]["dimension"], 4);
        assert!((response["result"]["eigenvalues"][0].as_f64().unwrap() + 0.75).abs() < 1.0e-12);
    }
}
