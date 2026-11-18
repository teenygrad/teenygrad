//! `teeny-llm-core`: the transport-independent core of teenygrad's LLM inference stack.
//!
//! Frontends ([`teeny-llm-server`], [`teeny-llm-chat`]) translate their own wire formats into a
//! [`GenerationRequest`] and drive an [`Engine`]. Nothing in this crate knows about HTTP or the
//! terminal.
//!
//! [`teeny-llm-server`]: https://docs.teenygrad.org/api/teenygrad/teeny_llm_server/
//! [`teeny-llm-chat`]: https://docs.teenygrad.org/api/teenygrad/teeny_llm_chat/

mod echo;
mod engine;
mod error;
mod types;

pub use echo::EchoEngine;
pub use engine::{Engine, TokenStream};
pub use error::{Error, Result};
pub use types::{
    ChatMessage, Embeddings, FinishReason, Generation, GenerationChunk, GenerationRequest,
    ModelInfo, Prompt, Role, SamplingParams, Usage,
};
