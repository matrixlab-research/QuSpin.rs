use std::time::{SystemTime, UNIX_EPOCH};

use approx::assert_abs_diff_eq;
use quspin::Complex64;
use quspin::archive::{ARCHIVE_VERSION, load_operator_npz, save_operator_npz};
use quspin::operator::{LinearOperator, MatrixFormat, Operator};

#[test]
fn versioned_npz_round_trip_preserves_complex_values_and_requested_format() {
    assert_eq!(ARCHIVE_VERSION, 1);
    let operator = Operator::from_dense(
        2,
        2,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.25, -0.5),
            Complex64::new(0.25, 0.5),
            Complex64::new(-2.0, 0.0),
        ],
    )
    .unwrap();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path =
        std::env::temp_dir().join(format!("quspin-archive-{}-{nonce}.npz", std::process::id()));
    save_operator_npz(&operator, &path).unwrap();
    let loaded = load_operator_npz(&path, MatrixFormat::Csc).unwrap();
    assert_eq!(loaded.format(), MatrixFormat::Csc);
    for (actual, expected) in loaded.to_dense().iter().zip(operator.to_dense()) {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-15);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-15);
    }
    std::fs::remove_file(path).unwrap();
}
