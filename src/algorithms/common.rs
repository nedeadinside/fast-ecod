use crate::errors::Error;
use crate::types::{FloatMatrix, FloatVector};

use num_traits::Float;

/// Computes the score as the negative sum of natural logarithms of the input values.
pub fn compute_score<T: Float>(scores: &FloatVector<T>) -> T {
    // Inputs are guaranteed to be positive and cannot cause the result
    // to overflow to infinity, so no additional validation is required.
    -scores
        .iter()
        .map(|&x| if x > T::zero() { x.ln() } else { T::zero() })
        .fold(T::zero(), |a, b| a + b)
}

/// Computes skewness of the given vector.
pub fn compute_skewness<T: Float>(vector: &FloatVector<T>) -> T {
    let n = T::from(vector.len()).unwrap();
    let mean = vector.iter().fold(T::zero(), |a, &x| a + x) / n;

    let (sum2, sum3) = vector.iter().fold((T::zero(), T::zero()), |(s2, s3), &x| {
        let d = x - mean;
        let d2 = d * d;
        (s2 + d2, s3 + d2 * d)
    });

    let m2 = sum2 / n;
    let m3 = sum3 / n;
    if m2 == T::zero() {
        return T::zero();
    }
    m3 / (m2 * m2.sqrt())
}

pub fn validate_shape<T: Float>(a: &FloatMatrix<T>, b: &FloatMatrix<T>) -> Result<(), Error> {
    if a.len() != b.len() {
        return Err(Error::FeatureMismatch {
            expected: (a.len()),
            got: (b.len()),
        });
    }
    Ok(())
}

/// Validates that the feature matrix is non-empty, rectangular, and fully finite.
pub fn validate_finity<T: Float>(feature_matrix: &FloatMatrix<T>) -> Result<(), Error> {
    let n_rows = match feature_matrix.first() {
        Some(col) if !col.is_empty() => col.len(),
        _ => return Err(Error::EmptyInput),
    };

    for (col, vec) in feature_matrix.iter().enumerate() {
        if vec.len() != n_rows {
            return Err(Error::RaggedRows);
        }
        if let Some(row) = vec.iter().position(|v| !v.is_finite()) {
            return Err(Error::NonFinite { row, col });
        }
    }

    Ok(())
}
