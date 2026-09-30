use crate::errors::Error;
use num_traits::Float;

/// may be i will add another types later
pub type FloatMatrix<T> = Vec<Vec<T>>;
pub type FloatVector<T> = Vec<T>;

/// Base trait to initialize model parameters
pub trait Fit<T: Float> {
    type Model;
    fn fit(&self, x: &FloatMatrix<T>) -> Result<Self::Model, Error>;
}

/// Base trait to get predictions from model
pub trait Predict<T: Float> {
    fn decision_function(&self, x: &FloatMatrix<T>) -> Result<FloatVector<T>, Error>;
}
