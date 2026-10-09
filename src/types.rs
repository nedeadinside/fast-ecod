use crate::errors::Error;
use num_traits::Float;
use pyo3::prelude::*;

/// Enum for each ECOD decision function type
#[pyclass(eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub enum ECODScoreMethod {
    RIGHT,
    LEFT,
    AUTO,
    MAX,
}

pub type FloatMatrix<T> = Vec<Vec<T>>;
pub type FloatVector<T> = Vec<T>;

/// Base trait to initialize model parameters
pub trait Fit<T: Float> {
    type Model;
    fn fit(&self, x: &FloatMatrix<T>) -> Result<Self::Model, Error>;
}

/// Base trait to get predictions from model
pub trait Predict<T: Float> {
    fn decision_function(
        &self,
        x: &FloatMatrix<T>,
        method: ECODScoreMethod,
    ) -> Result<FloatVector<T>, Error>;
}
