#[derive(Debug)]
pub enum Error {
    /// Input matrix X is empty.
    EmptyInput,
    /// Rows of X have different lengths.
    RaggedRows,
    /// Number of features passed to predict does not match the number seen during fit.
    FeatureMismatch {
        /// Number of features seen during fit.
        expected: usize,
        /// Number of features passed to predict.
        got: usize,
    },
    /// Input contains a non-finite value.
    NonFinite {
        /// Row index of the value.
        row: usize,
        /// Column index of the value.
        col: usize,
    },
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::EmptyInput => write!(f, "input is empty"),
            Error::RaggedRows => {
                write!(f, "feature matrix has different sizes of rows")
            }
            Error::FeatureMismatch { expected, got } => {
                write!(f, "model fitted on {expected} features, got {got}")
            }
            Error::NonFinite { row, col } => write!(f, "non-finite value at ({row}, {col})"),
        }
    }
}

impl std::error::Error for Error {}
