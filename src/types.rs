use crate::errors::Error;

/// may be i will add another types later
pub type FloatMatrix = Vec<Vec<f32>>;
pub type FloatVector = Vec<f32>;

/// Base trait to initialize model parameters
pub trait Fit {
    type Model;
    fn fit(&self, x: &FloatMatrix) -> Result<Self::Model, Error>;
}

/// Base trait to get predictions from model
pub trait Predict {
    fn decision_function(&self, x: &FloatMatrix) -> Result<FloatVector, Error>;
}
