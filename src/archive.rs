use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use num_complex::Complex64;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use crate::operator::{LinearOperator, MatrixFormat, Operator, materialize_dense};
use crate::{QuSpinError, Result};

pub const ARCHIVE_VERSION: u8 = 1;

fn archive_error(error: impl std::fmt::Display) -> QuSpinError {
    QuSpinError::Archive(error.to_string())
}

fn npy_header(descriptor: &str, shape: &[usize]) -> Result<Vec<u8>> {
    let shape = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!(
            "({})",
            shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let dictionary =
        format!("{{'descr': '{descriptor}', 'fortran_order': False, 'shape': {shape}, }}");
    let padding = (16 - ((10 + dictionary.len() + 1) % 16)) % 16;
    let header_length = dictionary
        .len()
        .checked_add(padding)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| QuSpinError::Archive("NPY header length overflow".into()))?;
    let header_length = u16::try_from(header_length)
        .map_err(|_| QuSpinError::Archive("NPY v1 header is too long".into()))?;
    let mut output = Vec::with_capacity(10 + usize::from(header_length));
    output.extend_from_slice(b"\x93NUMPY");
    output.extend_from_slice(&[1, 0]);
    output.extend_from_slice(&header_length.to_le_bytes());
    output.extend_from_slice(dictionary.as_bytes());
    output.extend(std::iter::repeat_n(b' ', padding));
    output.push(b'\n');
    Ok(output)
}

fn write_u8_npy(writer: &mut impl Write, values: &[u8]) -> Result<()> {
    writer
        .write_all(&npy_header("|u1", &[values.len()])?)
        .map_err(archive_error)?;
    writer.write_all(values).map_err(archive_error)
}

fn write_complex_npy(
    writer: &mut impl Write,
    rows: usize,
    columns: usize,
    values: &[Complex64],
) -> Result<()> {
    if values.len() != rows.saturating_mul(columns) {
        return Err(QuSpinError::DimensionMismatch(
            "archive matrix values do not match its shape".into(),
        ));
    }
    writer
        .write_all(&npy_header("<c16", &[rows, columns])?)
        .map_err(archive_error)?;
    for value in values {
        writer
            .write_all(&value.re.to_le_bytes())
            .and_then(|()| writer.write_all(&value.im.to_le_bytes()))
            .map_err(archive_error)?;
    }
    Ok(())
}

fn parse_npy_header(reader: &mut impl Read) -> Result<(String, Vec<usize>)> {
    let mut magic = [0_u8; 6];
    reader.read_exact(&mut magic).map_err(archive_error)?;
    if &magic != b"\x93NUMPY" {
        return Err(QuSpinError::Archive("invalid NPY magic".into()));
    }
    let mut version = [0_u8; 2];
    reader.read_exact(&mut version).map_err(archive_error)?;
    let header_length = match version[0] {
        1 => {
            let mut bytes = [0_u8; 2];
            reader.read_exact(&mut bytes).map_err(archive_error)?;
            usize::from(u16::from_le_bytes(bytes))
        }
        2 | 3 => {
            let mut bytes = [0_u8; 4];
            reader.read_exact(&mut bytes).map_err(archive_error)?;
            usize::try_from(u32::from_le_bytes(bytes))
                .map_err(|_| QuSpinError::Archive("NPY header is too large".into()))?
        }
        _ => return Err(QuSpinError::Archive("unsupported NPY version".into())),
    };
    let mut header = vec![0_u8; header_length];
    reader.read_exact(&mut header).map_err(archive_error)?;
    let header = std::str::from_utf8(&header).map_err(archive_error)?.trim();
    if header.contains("'fortran_order': True") || header.contains("\"fortran_order\": True") {
        return Err(QuSpinError::Archive(
            "Fortran-ordered NPY matrices are not supported".into(),
        ));
    }
    let descriptor_marker = if header.contains("'descr':") {
        "'descr':"
    } else {
        "\"descr\":"
    };
    let descriptor_tail = header
        .split_once(descriptor_marker)
        .ok_or_else(|| QuSpinError::Archive("NPY descriptor is missing".into()))?
        .1
        .trim_start();
    let quote = descriptor_tail
        .chars()
        .next()
        .filter(|character| *character == '\'' || *character == '"')
        .ok_or_else(|| QuSpinError::Archive("invalid NPY descriptor".into()))?;
    let descriptor = descriptor_tail[1..]
        .split_once(quote)
        .ok_or_else(|| QuSpinError::Archive("invalid NPY descriptor".into()))?
        .0
        .to_string();
    let shape_marker = if header.contains("'shape':") {
        "'shape':"
    } else {
        "\"shape\":"
    };
    let shape_tail = header
        .split_once(shape_marker)
        .ok_or_else(|| QuSpinError::Archive("NPY shape is missing".into()))?
        .1;
    let shape_text = shape_tail
        .split_once('(')
        .and_then(|(_, tail)| tail.split_once(')'))
        .map(|(shape, _)| shape)
        .ok_or_else(|| QuSpinError::Archive("invalid NPY shape".into()))?;
    let shape = shape_text
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.parse::<usize>().map_err(archive_error))
        .collect::<Result<Vec<_>>>()?;
    Ok((descriptor, shape))
}

fn read_version(reader: &mut impl Read) -> Result<u8> {
    let (descriptor, shape) = parse_npy_header(reader)?;
    if descriptor != "|u1" || shape != [1] {
        return Err(QuSpinError::Archive("invalid archive-version array".into()));
    }
    let mut version = [0_u8; 1];
    reader.read_exact(&mut version).map_err(archive_error)?;
    Ok(version[0])
}

fn read_complex_matrix(reader: &mut impl Read) -> Result<(usize, usize, Vec<Complex64>)> {
    let (descriptor, shape) = parse_npy_header(reader)?;
    if !matches!(descriptor.as_str(), "<c16" | "=c16" | "|c16") || shape.len() != 2 {
        return Err(QuSpinError::Archive(
            "matrix.npy must be a C-order complex128 matrix".into(),
        ));
    }
    let count = shape[0]
        .checked_mul(shape[1])
        .ok_or_else(|| QuSpinError::Archive("archive matrix size overflow".into()))?;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        let mut real = [0_u8; 8];
        let mut imaginary = [0_u8; 8];
        reader.read_exact(&mut real).map_err(archive_error)?;
        reader.read_exact(&mut imaginary).map_err(archive_error)?;
        values.push(Complex64::new(
            f64::from_le_bytes(real),
            f64::from_le_bytes(imaginary),
        ));
    }
    Ok((shape[0], shape[1], values))
}

/// Save a versioned, pickle-free NPZ archive readable as `np.load(path)["matrix"]`.
pub fn save_operator_npz(
    operator: &(impl LinearOperator + ?Sized),
    path: impl AsRef<Path>,
) -> Result<()> {
    let shape = operator.shape();
    let values = materialize_dense(operator)?;
    let file = File::create(path).map_err(archive_error)?;
    let mut archive = ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    archive
        .start_file("archive_version.npy", options)
        .map_err(archive_error)?;
    write_u8_npy(&mut archive, &[ARCHIVE_VERSION])?;
    archive
        .start_file("matrix.npy", options)
        .map_err(archive_error)?;
    write_complex_npy(&mut archive, shape.0, shape.1, &values)?;
    archive.finish().map_err(archive_error)?;
    Ok(())
}

/// Load a pickle-free NPZ produced by Rust or NumPy `savez`/`savez_compressed`.
pub fn load_operator_npz(path: impl AsRef<Path>, format: MatrixFormat) -> Result<Operator> {
    let file = File::open(path).map_err(archive_error)?;
    let mut archive = ZipArchive::new(file).map_err(archive_error)?;
    if let Ok(mut version_entry) = archive.by_name("archive_version.npy") {
        let version = read_version(&mut version_entry)?;
        if version != ARCHIVE_VERSION {
            return Err(QuSpinError::Archive(format!(
                "unsupported operator archive version {version}"
            )));
        }
    }
    let mut matrix = archive.by_name("matrix.npy").map_err(archive_error)?;
    let (rows, columns, values) = read_complex_matrix(&mut matrix)?;
    Operator::from_dense(rows, columns, values)?.converted(format)
}

pub use load_operator_npz as load_zip;
pub use save_operator_npz as save_zip;
