#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

pub mod archive;
pub mod basis;
pub mod block;
pub mod dynamics;
pub mod error;
pub mod measure;
pub mod operator;
pub mod solve;
pub mod workflow;

pub use error::{QuSpinError, Result};
pub use num_complex::Complex64;

/// Crate version used by verification adapters.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
