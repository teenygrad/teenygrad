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

//! `teeny-api`: teenygrad's inference served over an OpenAI-compatible HTTP API.
//!
//! [`Engine`] is what the routes drive, over the APIs of every domain this build includes;
//! [`ModelEngine`] implements it over [`ServedModel`]s, each holding the API handles a
//! domain crate's model implements, and routes each request to the model and API it
//! names. [`openai`] holds the wire types (a subset of the OpenAI OpenAPI spec) and
//! [`router`] builds the axum [`Router`](axum::Router) that serves them.
//!
//! Each domain is a cargo feature that brings in its crate and endpoints:
//!
//! | Feature | Crate | Endpoints |
//! |---|---|---|
//! | `llm` (default) | [`teeny-llm`] | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/moderations` |
//! | `audio` | [`teeny-llm`] (for now) | `/v1/audio/transcriptions`, `/v1/audio/translations`, `/v1/audio/speech` |
//! | `image` | [`teeny-llm`] (for now) | `/v1/images/generations`, `/v1/images/edits`, `/v1/images/variations` |
//!
//! `/v1/models` and `/health` are always served. The models themselves are features too
//! (`functiongemma3`), with the backend they run on (`cuda`, `riscv`, `qemu`).
//!
//! [`teeny-llm`]: https://docs.teenygrad.org/api/teenygrad/teeny_llm/

mod engine;
mod error;
mod model_engine;
pub mod openai;
mod routes;

pub use engine::{Engine, ModelInfo};
pub use error::{Error, Result};
pub use model_engine::{ModelEngine, ServedModel};
pub use routes::router;
