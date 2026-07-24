use qmbed::Complex64;
use qmbed::archive::{load_operator_npz, save_operator_npz};
use qmbed::operator::{MatrixFormat, Operator};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args().collect();
    match arguments.as_slice() {
        [_, mode, path] if mode == "write" => {
            let operator = Operator::from_dense(
                2,
                2,
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.25, -0.5),
                    Complex64::new(0.25, 0.5),
                    Complex64::new(-2.0, 0.0),
                ],
            )?;
            save_operator_npz(&operator, path)?;
        }
        [_, mode, path] if mode == "read" => {
            let operator = load_operator_npz(path, MatrixFormat::Dense)?;
            for value in operator.to_dense() {
                println!("{:.17},{:.17}", value.re, value.im);
            }
        }
        _ => return Err("usage: archive_interop <write|read> <path>".into()),
    }
    Ok(())
}
