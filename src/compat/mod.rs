//! Compatibility namespaces for APIs that predate QMBED's native surface.

/// QuSpin-derived Rust spellings retained during the QMBED migration.
///
/// New code should import QMBED modules directly. This namespace is an adapter:
/// it does not contain a second implementation or a model-specific execution
/// path.
pub mod quspin {
    pub use crate::archive;
    pub use crate::basis;
    pub use crate::block;
    pub use crate::dynamics;
    pub use crate::error::QuSpinError;
    pub use crate::measure;
    pub use crate::operator;
    pub use crate::solve;
    pub use crate::workflow;
    pub use crate::{Complex64, Result, VERSION};
}
