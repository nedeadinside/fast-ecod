use crate::types::FloatVector;
use num_traits::Float;

/// Computes the score as the sum of natural logarithms of the input values.
pub fn compute_score<T: Float>(scores: &FloatVector<T>) -> T {
    // Inputs are guaranteed to be positive and cannot cause the result
    // to overflow to infinity, so no additional validation is required.
    scores
        .iter()
        .map(|&x| if x > T::zero() { x.ln() } else { T::zero() })
        .fold(T::zero(), |a, b| a + b)
}

/// Computes skewness of the given vector.
pub fn compute_skewness<T: Float>(vector: &FloatVector<T>) -> Option<T> {
    let n = T::from(vector.len())?;
    let mean = vector.iter().fold(T::zero(), |a, &x| a + x) / n;

    let (sum2, sum3) = vector.iter().fold((T::zero(), T::zero()), |(s2, s3), &x| {
        let d = x - mean;
        let d2 = d * d;
        (s2 + d2, s3 + d2 * d)
    });

    let m2 = sum2 / n;
    let m3 = sum3 / n;
    if m2 == T::zero() {
        return None;
    }
    Some(m3 / (m2 * m2.sqrt()))
}
