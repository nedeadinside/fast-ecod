use pyo3::prelude::*;

mod algorithms;
mod errors;
mod types;

/// Rust core of fast_ecod.
#[pymodule]
mod _core {}
