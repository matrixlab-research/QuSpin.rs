#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

/// Crate version used by verification adapters.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
