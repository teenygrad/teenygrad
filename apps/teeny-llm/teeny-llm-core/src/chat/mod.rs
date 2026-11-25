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

//! Chat completions: the interface models implement to be served.
//!
//! A model implements [`ChatModel`]: it reports its [`Capabilities`] and answers a
//! [`ChatRequest`] (the whole of OpenAI's chat completions request, as Rust types) with a
//! [`ChatStream`] of [`ChatEvent`]s. [`validate`] checks each request against the model's
//! capabilities first, so what a model cannot do is refused with an error naming the
//! parameter rather than ignored. [`collect`] puts a stream together into a
//! [`ChatCompletion`], and [`ModelEngine`](crate::ModelEngine) serves chat models to
//! frontends as an [`Engine`](crate::Engine). [`EchoChatModel`] is the reference
//! implementation.
//!
//! These types carry no wire format: frontends map their own (the server's OpenAI JSON)
//! onto them.

mod capabilities;
mod echo;
mod model;
mod request;
mod response;

pub use capabilities::{
    Capabilities, InputModalities, MAX_STOP_STRINGS, MAX_TOP_LOGPROBS, SamplingSupport,
    StructuredOutput, ToolSupport, validate,
};
pub use echo::EchoChatModel;
pub use model::ChatModel;
pub use request::{
    AllowedMode, AssistantMessage, AudioFormat, AudioOutput, AudioReply, ChatRequest, ContentPart,
    CustomFormat, GrammarSyntax, ImageDetail, Logprobs, Message, Modality, ReasoningEffort,
    ResponseFormat, Sampling, Tool, ToolCall, ToolCallKind, ToolChoice, ToolKind, Verbosity,
};
pub use response::{
    AudioDelta, ChatCompletion, ChatEvent, ChatFinishReason, ChatStream, ChatUsage, Choice, Delta,
    TokenLogprob, ToolCallDelta, TopLogprob, collect,
};
