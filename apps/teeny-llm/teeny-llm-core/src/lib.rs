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
//! Frontends ([`teeny-llm-server`], [`teeny-llm-chat`]) translate their own wire formats into a
//! [`GenerationRequest`] and drive an [`Engine`]. Nothing in this crate knows about HTTP or the
//! terminal.
//!
//! Model implementations live in `models`, each behind a cargo feature named after it
//! (`functiongemma`), with the kernels they share in `kernels`. Pick the backend with
//! `cuda`, `riscv` or `qemu`; the defaults are `functiongemma` and `cuda`.
//!
//! [`teeny-llm-server`]: https://docs.teenygrad.org/api/teenygrad/teeny_llm_server/
//! [`teeny-llm-chat`]: https://docs.teenygrad.org/api/teenygrad/teeny_llm_chat/

mod echo;
mod engine;
mod error;
mod types;

#[cfg(feature = "kernels")]
pub mod kernels;
#[cfg(any(feature = "functiongemma", feature = "inkling2"))]
pub mod models;

pub use echo::EchoEngine;
pub use engine::{Engine, TokenStream};
pub use error::{Error, Result};
pub use types::{
    ChatMessage, Embeddings, FinishReason, Generation, GenerationChunk, GenerationRequest,
    ModelInfo, Prompt, Role, SamplingParams, Usage,
};
