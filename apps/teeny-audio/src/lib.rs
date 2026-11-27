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

//! `teeny-audio`: the transport-independent audio APIs of teenygrad's inference stack.
//!
//! One module per API, each with its request and response types, the capabilities a model
//! declares, `validate` (refusing what a model cannot do, by parameter), the trait models
//! implement, and an echo model as its reference implementation:
//!
//! - [`transcription`]: speech to text ([`transcription::TranscriptionModel`]), and
//!   translation into English;
//! - [`speech`]: text to speech ([`speech::SpeechModel`]).
//!
//! [`codec`] decodes uploads into [`Pcm`] samples (WAV so far), mixed and resampled for the
//! model, and encodes samples into response formats (WAV, raw PCM), so no model repeats
//! either. Nothing here knows about HTTP: [`teeny-api`] serves these APIs with its `audio`
//! feature.
//!
//! [`teeny-api`]: https://docs.teenygrad.org/api/teenygrad/teeny_api/

pub mod codec;
mod error;
pub mod speech;
pub mod transcription;

pub use codec::{AudioContainer, AudioEncoding, Pcm};
pub use error::{Error, Result};

/// What a model is, as a frontend lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    pub id: String,
    /// Unix timestamp (seconds) the model was loaded/created.
    pub created: u64,
    pub owned_by: String,
}
