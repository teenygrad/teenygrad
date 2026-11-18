/// Errors produced by an [`Engine`](crate::Engine).
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("model `{requested}` is not loaded; available models: {available:?}")]
    ModelNotFound {
        requested: String,
        available: Vec<String>,
    },

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("engine `{engine}` does not support {operation}")]
    Unsupported {
        engine: &'static str,
        operation: &'static str,
    },

    #[error("inference failed: {0}")]
    Inference(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
