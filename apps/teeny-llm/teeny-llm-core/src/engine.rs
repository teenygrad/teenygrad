use std::pin::Pin;

use futures::{Stream, StreamExt};

use crate::{Embeddings, Error, Generation, GenerationChunk, GenerationRequest, ModelInfo, Result};

/// A stream of generated text increments; see [`GenerationChunk`].
pub type TokenStream = Pin<Box<dyn Stream<Item = Result<GenerationChunk>> + Send>>;

/// An inference backend.
///
/// Object-safe so frontends can hold an `Arc<dyn Engine>` without being generic over the backend.
pub trait Engine: Send + Sync + 'static {
    /// A short, stable name for diagnostics (e.g. `"echo"`).
    fn name(&self) -> &'static str;

    /// The models this engine can serve.
    fn models(&self) -> Vec<ModelInfo>;

    /// Starts a generation and returns its incremental output.
    ///
    /// Validation errors (unknown model, bad parameters) are returned here, before streaming
    /// starts, so frontends can map them to a proper error response.
    fn generate_stream(&self, request: GenerationRequest) -> Result<TokenStream>;

    /// Embeds `inputs` with `model`. Engines without an embedding model keep the default.
    fn embed(&self, model: &str, inputs: &[String]) -> Result<Embeddings> {
        let _ = (model, inputs);
        Err(Error::Unsupported {
            engine: self.name(),
            operation: "embeddings",
        })
    }

    /// Looks up a served model by id.
    fn model(&self, id: &str) -> Result<ModelInfo> {
        let models = self.models();
        match models.iter().find(|m| m.id == id) {
            Some(m) => Ok(m.clone()),
            None => Err(Error::ModelNotFound {
                requested: id.to_owned(),
                available: models.into_iter().map(|m| m.id).collect(),
            }),
        }
    }
}

impl dyn Engine {
    /// Runs a generation to completion, concatenating the streamed chunks.
    pub async fn generate(&self, request: GenerationRequest) -> Result<Generation> {
        let mut stream = self.generate_stream(request)?;
        let mut text = String::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            text.push_str(&chunk.text);
            if let Some(finish_reason) = chunk.finish_reason {
                return Ok(Generation {
                    text,
                    finish_reason,
                    usage: chunk.usage.unwrap_or_default(),
                });
            }
        }
        Err(Error::Inference(format!(
            "engine `{}` ended the stream without a finish reason after {} bytes of output",
            self.name(),
            text.len()
        )))
    }
}
