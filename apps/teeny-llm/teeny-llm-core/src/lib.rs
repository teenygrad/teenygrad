/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! `teeny-llm-core`: the transport-independent core of teenygrad's LLM inference stack.
//!
//! Frontends ([`teeny-llm-server`], [`teeny-llm-chat`]) translate their own wire formats into
//! the requests of each API ([`chat`], [`completion`], [`embedding`], [`transcription`],
//! [`speech`], [`moderation`]) and drive an
//! [`Engine`]. Models implement the API traits ([`chat::ChatModel`],
//! [`completion::CompletionModel`], [`embedding::EmbeddingModel`],
//! [`transcription::TranscriptionModel`], [`speech::SpeechModel`],
//! [`moderation::ModerationModel`]);
//! [`models::Model::load`] loads one as a [`models::LoadedModel`], and [`ModelEngine`]
//! serves loaded models as an [`Engine`]. Each API has an echo model, its reference
//! implementation and a fixture for frontends. Nothing in this crate knows about
//! HTTP or the terminal.
//!
//! Model implementations live in `models`, each behind a cargo feature named for the model
//! and its version (`functiongemma3` for `models::functiongemma`), with the kernels they
//! share in `kernels`; [`models::Model`] lists the ones a build can run, for command lines
//! to take as `--model`. Pick the backend with `cuda`, `riscv` or `qemu`; the defaults are
//! `functiongemma3` and `cuda`.
//!
//! [`teeny-llm-server`]: https://docs.teenygrad.org/api/teenygrad/teeny_llm_server/
//! [`teeny-llm-chat`]: https://docs.teenygrad.org/api/teenygrad/teeny_llm_chat/

pub mod audio;
pub mod chat;
pub mod completion;
pub mod embedding;
mod engine;
mod error;
mod model_engine;
pub mod speech;
pub mod tokenizer;
pub mod transcription;
mod types;

#[cfg(feature = "kernels")]
pub mod kernels;
pub mod models;
pub mod moderation;

pub use embedding::Embeddings;
pub use engine::Engine;
pub use error::{Error, Result};
pub use model_engine::ModelEngine;
pub use types::{ModelInfo, Usage};
