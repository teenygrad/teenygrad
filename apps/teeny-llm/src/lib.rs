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

//! `teeny-llm`: the transport-independent core of teenygrad's LLM inference stack.
//!
//! Frontends (such as [`teeny-api`]) translate their own wire formats into
//! the requests of each API ([`chat`], [`completion`], [`embedding`], [`moderation`]).
//! Models implement the API traits ([`chat::ChatModel`], [`completion::CompletionModel`],
//! [`embedding::EmbeddingModel`], [`moderation::ModerationModel`]), and [`models::Model::load`]
//! loads one as a [`models::LoadedModel`], exposing the APIs it serves; each request is
//! checked with its API's `validate` before the model sees it. Each API has an echo
//! model, its reference implementation and a fixture for frontends. Nothing in this crate
//! knows about HTTP or the terminal: `teeny-api` routes requests to the models it serves.
//!
//! Model implementations live in `models`, each behind a cargo feature named for the model
//! and its version (`functiongemma3` for `models::functiongemma`), with the kernels they
//! share in `kernels`; [`models::Model`] lists the ones a build can run, for command lines
//! to take as `--model`. Pick the backend with `cuda`, `riscv` or `qemu`; the defaults are
//! `functiongemma3` and `cuda`.
//!
//! [`teeny-api`]: https://docs.teenygrad.org/api/teenygrad/teeny_api/

pub mod chat;
pub mod completion;
pub mod embedding;
mod error;
pub mod tokenizer;
mod types;

#[cfg(feature = "kernels")]
pub mod kernels;
pub mod models;
pub mod moderation;

pub use embedding::Embeddings;
pub use error::{Error, Result};
pub use types::{ModelInfo, Usage};
