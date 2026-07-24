use qmbed::compat;
use qmbed::{QmbedError, QuSpinError};

#[test]
fn qmbed_is_the_native_crate_identity() {
    assert_eq!(qmbed::VERSION, env!("CARGO_PKG_VERSION"));
    let error = QmbedError::InvalidOptions("example".into());
    assert!(matches!(error, QmbedError::InvalidOptions(_)));
}

#[test]
fn quspin_error_spelling_is_a_compatibility_alias() {
    let error: QuSpinError = QmbedError::StateNotInBasis;
    assert!(matches!(
        error,
        compat::quspin::QuSpinError::StateNotInBasis
    ));
    assert_eq!(compat::quspin::VERSION, qmbed::VERSION);
}
