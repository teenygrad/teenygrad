//! `teeny-llm-server`: an OpenAI-compatible HTTP API over a [`teeny_llm_core::Engine`].
//!
//! [`openai`] holds the wire types (a subset of the OpenAI OpenAPI spec) and [`router`] builds the
//! axum [`Router`](axum::Router) that serves them.

pub mod openai;
mod routes;

pub use routes::router;
