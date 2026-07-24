//! Stable, language-neutral entry point shared by the Python and Julia layers.

use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};

use qmbed::basis::{
    Basis, BosonBasis1D, PackedBasis, SpinBasis1D, SpinfulFermionBasis1D, SpinlessFermionBasis1D,
};
use qmbed::interop::PackedEdModel;
use qmbed::operator::{
    AssemblyChecks, Coupling, LocalOperator, MatrixFormat, OpProduct, OperatorSpec,
};
use qmbed::solve::{EighOptions, EigshOptions, SpectrumTarget};
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
#[serde(tag = "operation", rename_all = "snake_case")]
enum CommandRequest {
    DescribeBasis {
        basis: BasisRequest,
    },
    Materialize {
        basis: BasisRequest,
        terms: Vec<TermRequest>,
        site_permutation: Option<Vec<usize>>,
        #[serde(default)]
        format: StorageFormat,
        #[serde(default)]
        checks: ChecksRequest,
    },
    Eigh {
        basis: BasisRequest,
        terms: Vec<TermRequest>,
        site_permutation: Option<Vec<usize>>,
        #[serde(default)]
        eigenvectors: bool,
        #[serde(default)]
        checks: ChecksRequest,
    },
    Eigsh {
        basis: BasisRequest,
        terms: Vec<TermRequest>,
        site_permutation: Option<Vec<usize>>,
        #[serde(default)]
        format: StorageFormat,
        solver: SolverRequest,
        #[serde(default)]
        checks: ChecksRequest,
    },
}

#[derive(Debug, Default, Deserialize)]
struct ChecksRequest {
    hermiticity: Option<bool>,
    particle_conservation: Option<bool>,
    symmetry_compatibility: Option<bool>,
}

impl From<ChecksRequest> for AssemblyChecks {
    fn from(checks: ChecksRequest) -> Self {
        Self {
            hermiticity: checks.hermiticity.unwrap_or(true),
            particle_conservation: checks.particle_conservation.unwrap_or(true),
            symmetry_compatibility: checks.symmetry_compatibility.unwrap_or(true),
        }
    }
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
        #[serde(default)]
        reverse: bool,
    },
    Boson {
        sites: usize,
        particles: Option<usize>,
        states_per_site: usize,
        #[serde(default)]
        reverse: bool,
    },
    SpinlessFermion {
        sites: usize,
        particles: Option<usize>,
        momentum: Option<i32>,
        #[serde(default)]
        reverse: bool,
    },
    SpinfulFermion {
        sites: usize,
        particles_up: Option<usize>,
        particles_down: Option<usize>,
        #[serde(default)]
        reverse: bool,
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

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum StorageFormat {
    Dense,
    #[default]
    Csc,
    Csr,
    Dia,
    MatrixFree,
}

#[derive(Debug, Serialize)]
struct MatrixEntry {
    row: usize,
    column: usize,
    value: [f64; 2],
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum CommandResult {
    Basis {
        dimension: usize,
        states: Vec<String>,
    },
    Operator {
        shape: [usize; 2],
        format: StorageFormat,
        entries: Vec<MatrixEntry>,
    },
    Eigensystem {
        dimension: usize,
        eigenvalues: Vec<f64>,
        residuals: Vec<f64>,
        iterations: usize,
        converged: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        eigenvectors: Option<Vec<Vec<[f64; 2]>>>,
    },
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

#[derive(Debug, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum CommandResponse {
    Ok { result: CommandResult },
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

pub fn run_command_json(request: &str) -> String {
    let response = serde_json::from_str::<CommandRequest>(request)
        .map_err(|error| QmbedError::InvalidOptions(format!("invalid binding command: {error}")))
        .and_then(execute_command)
        .map_or_else(
            |error| CommandResponse::Error {
                error: error.to_string(),
            },
            |result| CommandResponse::Ok { result },
        );
    serde_json::to_string(&response).unwrap_or_else(|error| {
        format!(r#"{{"status":"error","error":"response serialization failed: {error}"}}"#)
    })
}

fn execute(request: SolveRequest) -> Result<SolveResult> {
    let model = build_model(&request.basis, request.terms, AssemblyChecks::all(), None)?;
    solve_model(&model, request.format, &request.solver)
}

fn build_basis(request: &BasisRequest) -> Result<PackedBasis> {
    match request {
        BasisRequest::Spin {
            sites,
            spin_twice,
            up,
            momentum,
            parity,
            pauli,
            reverse,
        } => {
            let mut builder = SpinBasis1D::builder(*sites)
                .spin_twice(*spin_twice)
                .pauli(*pauli);
            if let Some(up) = up {
                builder = builder.up(*up);
            }
            if let Some(momentum) = momentum {
                builder = builder.momentum(*momentum);
            }
            if let Some(parity) = parity {
                builder = builder.parity(*parity);
            }
            Ok(ordered_basis(builder.build()?.into(), *reverse))
        }
        BasisRequest::Boson {
            sites,
            particles,
            states_per_site,
            reverse,
        } => {
            let mut builder = BosonBasis1D::builder(*sites, *states_per_site);
            if let Some(particles) = particles {
                builder = builder.particles(*particles);
            }
            Ok(ordered_basis(builder.build()?.into(), *reverse))
        }
        BasisRequest::SpinlessFermion {
            sites,
            particles,
            momentum,
            reverse,
        } => {
            let mut builder = SpinlessFermionBasis1D::builder(*sites);
            if let Some(particles) = particles {
                builder = builder.particles(*particles);
            }
            if let Some(momentum) = momentum {
                builder = builder.momentum(*momentum);
            }
            Ok(ordered_basis(builder.build()?.into(), *reverse))
        }
        BasisRequest::SpinfulFermion {
            sites,
            particles_up,
            particles_down,
            reverse,
        } => {
            let mut builder = SpinfulFermionBasis1D::builder(*sites);
            if let Some(particles) = particles_up {
                builder = builder.particles_up(*particles);
            }
            if let Some(particles) = particles_down {
                builder = builder.particles_down(*particles);
            }
            Ok(ordered_basis(builder.build()?.into(), *reverse))
        }
    }
}

fn ordered_basis(basis: PackedBasis, reverse: bool) -> PackedBasis {
    if reverse { basis.reversed() } else { basis }
}

fn build_model(
    basis: &BasisRequest,
    terms: Vec<TermRequest>,
    checks: AssemblyChecks,
    site_permutation: Option<Vec<usize>>,
) -> Result<PackedEdModel> {
    let terms = terms
        .into_iter()
        .map(typed_term)
        .collect::<Result<Vec<_>>>()?;
    let model = PackedEdModel::new(build_basis(basis)?, terms).with_checks(checks);
    match site_permutation {
        Some(permutation) => model.with_site_permutation(&permutation),
        None => Ok(model),
    }
}

fn solve_model(
    model: &PackedEdModel,
    format: StorageFormat,
    solver: &SolverRequest,
) -> Result<SolveResult> {
    let include_vectors = solver.eigenvectors;
    let result = model.eigsh(
        format.into(),
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
        dimension: model.dimension(),
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

fn command_eigensystem(
    dimension: usize,
    result: qmbed::solve::Eigensystem,
    include_vectors: bool,
) -> CommandResult {
    CommandResult::Eigensystem {
        dimension,
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
    }
}

fn execute_command(request: CommandRequest) -> Result<CommandResult> {
    match request {
        CommandRequest::DescribeBasis { basis } => {
            let basis = build_basis(&basis)?;
            let states = (0..basis.len())
                .map(|index| basis.state(index).map(|state| state.to_string()))
                .collect::<Result<Vec<_>>>()?;
            Ok(CommandResult::Basis {
                dimension: basis.len(),
                states,
            })
        }
        CommandRequest::Materialize {
            basis,
            terms,
            site_permutation,
            format,
            checks,
        } => {
            let model = build_model(&basis, terms, checks.into(), site_permutation)?;
            let operator = model.materialize(format.into())?;
            let (rows, columns) = qmbed::operator::LinearOperator::shape(&operator);
            let entries = operator
                .triplets()
                .into_iter()
                .map(|(row, column, value)| MatrixEntry {
                    row,
                    column,
                    value: [value.re, value.im],
                })
                .collect();
            Ok(CommandResult::Operator {
                shape: [rows, columns],
                format,
                entries,
            })
        }
        CommandRequest::Eigh {
            basis,
            terms,
            site_permutation,
            eigenvectors,
            checks,
        } => {
            let model = build_model(&basis, terms, checks.into(), site_permutation)?;
            let result = model.eigh(EighOptions {
                return_eigenvectors: eigenvectors,
            })?;
            Ok(command_eigensystem(model.dimension(), result, eigenvectors))
        }
        CommandRequest::Eigsh {
            basis,
            terms,
            site_permutation,
            format,
            solver,
            checks,
        } => {
            let model = build_model(&basis, terms, checks.into(), site_permutation)?;
            let include_vectors = solver.eigenvectors;
            let result = model.eigsh(
                format.into(),
                EigshOptions {
                    eigenpairs: solver.eigenpairs,
                    target: solver.target.into(),
                    krylov_dimension: solver.krylov_dimension,
                    tolerance: solver.tolerance,
                    max_iterations: solver.max_iterations,
                    seed: solver.seed,
                },
            )?;
            Ok(command_eigensystem(
                model.dimension(),
                result,
                include_vectors,
            ))
        }
    }
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

/// Execute a reusable-model command encoded as JSON.
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
pub unsafe extern "C" fn qmbed_command_json(request: *const c_char) -> *mut c_char {
    let response = catch_unwind(AssertUnwindSafe(|| {
        if request.is_null() {
            return r#"{"status":"error","error":"request pointer is null"}"#.to_string();
        }
        // SAFETY: The caller contract requires a live NUL-terminated string.
        let request = unsafe { CStr::from_ptr(request) };
        match request.to_str() {
            Ok(request) => run_command_json(request),
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

    use super::{run_command_json, run_json};

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

    #[test]
    fn command_protocol_reuses_one_model_shape_for_basis_operator_and_eigh() {
        let basis = r#"{"kind":"boson","sites":1,"states_per_site":4}"#;
        let terms = r#"[
            {"product":{"local":["raising"]},"couplings":[{"coefficient":[1.0,0.0],"sites":[0]}]},
            {"product":{"local":["lowering"]},"couplings":[{"coefficient":[1.0,0.0],"sites":[0]}]},
            {"product":{"local":["number"]},"couplings":[{"coefficient":[0.25,0.0],"sites":[0]}]}
        ]"#;

        let describe = run_command_json(&format!(
            r#"{{"operation":"describe_basis","basis":{basis}}}"#
        ));
        let describe: Value = serde_json::from_str(&describe).unwrap();
        assert_eq!(describe["status"], "ok");
        assert_eq!(describe["result"]["dimension"], 4);
        assert_eq!(describe["result"]["states"][3], "3");

        let materialize = run_command_json(&format!(
            r#"{{"operation":"materialize","basis":{basis},"terms":{terms},"format":"csc"}}"#
        ));
        let materialize: Value = serde_json::from_str(&materialize).unwrap();
        assert_eq!(materialize["status"], "ok");
        assert_eq!(materialize["result"]["shape"], serde_json::json!([4, 4]));
        assert_eq!(
            materialize["result"]["entries"].as_array().unwrap().len(),
            9
        );

        let eigh = run_command_json(&format!(
            r#"{{"operation":"eigh","basis":{basis},"terms":{terms}}}"#
        ));
        let eigh: Value = serde_json::from_str(&eigh).unwrap();
        assert_eq!(eigh["status"], "ok");
        assert_eq!(eigh["result"]["eigenvalues"].as_array().unwrap().len(), 4);
        assert!(eigh["result"].get("eigenvectors").is_none());
    }
}
