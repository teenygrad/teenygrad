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

//! A chat completion request: the model-facing form of OpenAI's
//! `POST /v1/chat/completions` body.

use serde_json::Value;

/// One chat completion request.
///
/// Every field of the OpenAI request a model can act on has a place here. Fields that only
/// concern the service in front of the model (`store`, `service_tier`, `metadata` for
/// dashboards) are carried in [`ChatRequest::user`] and [`ChatRequest::metadata`] or stay in
/// the frontend. `Default` gives an empty request (model `""`, one choice), to be filled in
/// with struct update syntax.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatRequest {
    /// The model id, as listed by [`ChatModel::info`](super::ChatModel::info).
    pub model: String,
    /// The conversation so far, oldest first.
    pub messages: Vec<Message>,
    /// The tools the model may call.
    pub tools: Vec<Tool>,
    /// Whether, and which, tool the model must call.
    pub tool_choice: ToolChoice,
    /// Whether the model may call more than one tool in a turn. `None` is the model's default.
    pub parallel_tool_calls: Option<bool>,
    /// The shape the reply must take.
    pub response_format: ResponseFormat,
    /// Decoding controls.
    pub sampling: Sampling,
    /// Most tokens to generate per choice, reasoning tokens included. OpenAI's
    /// `max_completion_tokens`, or the legacy `max_tokens`.
    pub max_completion_tokens: Option<u32>,
    /// How many choices to generate. OpenAI's `n`; 1 when not given.
    pub n: u32,
    /// Per-token log probabilities to return with the reply, if any.
    pub logprobs: Option<Logprobs>,
    /// How hard a reasoning model thinks before answering.
    pub reasoning_effort: Option<ReasoningEffort>,
    /// How much a model says, for models that take the hint.
    pub verbosity: Option<Verbosity>,
    /// The output kinds wanted. Empty means text only.
    pub modalities: Vec<Modality>,
    /// The voice and format, when [`Modality::Audio`] is asked for.
    pub audio: Option<AudioOutput>,
    /// Text the reply is expected to mostly repeat (OpenAI's predicted outputs), which a
    /// model can use to decode faster.
    pub prediction: Option<String>,
    /// Whether a stream ends with the request's usage. OpenAI's
    /// `stream_options.include_usage`; non-streamed replies always carry it.
    pub include_usage: bool,
    /// An opaque id for the end user, for abuse tracking.
    pub user: Option<String>,
    /// Key-value pairs attached to the request, passed through untouched.
    pub metadata: Vec<(String, String)>,
}

impl Default for ChatRequest {
    fn default() -> Self {
        Self {
            model: String::new(),
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: ToolChoice::default(),
            parallel_tool_calls: None,
            response_format: ResponseFormat::default(),
            sampling: Sampling::default(),
            max_completion_tokens: None,
            n: 1,
            logprobs: None,
            reasoning_effort: None,
            verbosity: None,
            modalities: Vec::new(),
            audio: None,
            prediction: None,
            include_usage: false,
            user: None,
            metadata: Vec::new(),
        }
    }
}

/// One turn of a conversation.
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    /// Instructions from the application developer. OpenAI's newer name for `system`;
    /// models that know only one treat them alike.
    Developer {
        content: String,
        name: Option<String>,
    },
    /// Instructions to the model.
    System {
        content: String,
        name: Option<String>,
    },
    /// The person (or program) the model is talking to.
    User {
        content: Vec<ContentPart>,
        name: Option<String>,
    },
    /// An earlier reply from the model.
    Assistant(AssistantMessage),
    /// The result of a tool call the model made.
    Tool {
        /// The [`ToolCall::id`] this answers.
        tool_call_id: String,
        content: String,
    },
}

impl Message {
    /// A user message of plain text.
    pub fn user(text: impl Into<String>) -> Self {
        Self::User {
            content: vec![ContentPart::Text(text.into())],
            name: None,
        }
    }

    /// A system message.
    pub fn system(text: impl Into<String>) -> Self {
        Self::System {
            content: text.into(),
            name: None,
        }
    }

    /// An assistant message of plain text.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::Assistant(AssistantMessage {
            content: Some(text.into()),
            ..AssistantMessage::default()
        })
    }

    /// The role name, as OpenAI spells it, for error messages.
    pub fn role(&self) -> &'static str {
        match self {
            Self::Developer { .. } => "developer",
            Self::System { .. } => "system",
            Self::User { .. } => "user",
            Self::Assistant(_) => "assistant",
            Self::Tool { .. } => "tool",
        }
    }
}

/// A reply from the model, as it appears in a conversation or a completed choice.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AssistantMessage {
    /// What the model said. `None` when the turn was only tool calls or a refusal.
    pub content: Option<String>,
    /// The model's explanation, when it declined to answer.
    pub refusal: Option<String>,
    /// The tools the model called.
    pub tool_calls: Vec<ToolCall>,
    /// The spoken reply, when audio output was asked for.
    pub audio: Option<AudioReply>,
    /// An optional participant name.
    pub name: Option<String>,
}

/// One part of a user message.
#[derive(Debug, Clone, PartialEq)]
pub enum ContentPart {
    Text(String),
    /// An image, by `https:` URL or `data:` URL.
    Image {
        url: String,
        detail: ImageDetail,
    },
    /// Spoken input.
    Audio {
        data: Vec<u8>,
        format: AudioFormat,
    },
    /// A document, either uploaded earlier (`file_id`) or inline.
    File {
        file_id: Option<String>,
        filename: Option<String>,
        data: Option<Vec<u8>>,
    },
}

impl ContentPart {
    /// The part kind, as OpenAI spells it, for error messages.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Text(_) => "text",
            Self::Image { .. } => "image_url",
            Self::Audio { .. } => "input_audio",
            Self::File { .. } => "file",
        }
    }
}

/// How closely a model should look at an image.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ImageDetail {
    #[default]
    Auto,
    Low,
    High,
}

/// The encoding of audio in or out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    Wav,
    Mp3,
    Flac,
    Opus,
    Aac,
    Pcm16,
}

/// A tool the model may call.
#[derive(Debug, Clone, PartialEq)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub kind: ToolKind,
}

/// What a tool takes as input.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolKind {
    /// A function called with JSON arguments.
    Function {
        /// A JSON Schema for the arguments. `None` means no arguments.
        parameters: Option<Value>,
        /// Whether the arguments must match `parameters` exactly.
        strict: bool,
    },
    /// A tool called with free-form text, optionally constrained by a grammar.
    Custom { format: CustomFormat },
}

/// The input a custom tool accepts.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum CustomFormat {
    /// Any text.
    #[default]
    Text,
    /// Text matching a grammar.
    Grammar {
        syntax: GrammarSyntax,
        definition: String,
    },
}

/// How a custom tool's grammar is written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrammarSyntax {
    Lark,
    Regex,
}

/// Whether, and which, tool the model must call.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum ToolChoice {
    /// The model decides. OpenAI's default when tools are given (and the only meaning
    /// without tools is "no call", which is what a model does with none declared).
    #[default]
    Auto,
    /// The model must not call a tool.
    None,
    /// The model must call at least one tool.
    Required,
    /// The model must call this tool.
    Named(String),
    /// The model may only call these tools, deciding (`Auto`) or having to (`Required`).
    Allowed {
        mode: AllowedMode,
        tools: Vec<String>,
    },
}

/// Whether a call is optional or required under [`ToolChoice::Allowed`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllowedMode {
    Auto,
    Required,
}

/// A tool call the model made.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    /// Unique within the conversation; a [`Message::Tool`] answers it by this id.
    pub id: String,
    pub name: String,
    /// Whether this called a function or a custom tool.
    pub kind: ToolCallKind,
    /// The JSON arguments of a function call, or the text input of a custom tool, exactly
    /// as the model wrote it. Models can write invalid JSON; callers parse it.
    pub input: String,
}

/// The kind of tool a [`ToolCall`] called.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ToolCallKind {
    #[default]
    Function,
    Custom,
}

/// The shape a reply must take.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum ResponseFormat {
    /// Free text.
    #[default]
    Text,
    /// Any valid JSON object.
    JsonObject,
    /// JSON matching a schema.
    JsonSchema {
        name: String,
        description: Option<String>,
        schema: Value,
        /// Whether the reply must match `schema` exactly.
        strict: bool,
    },
}

/// Decoding controls. `None` (or empty) means the model's default.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sampling {
    /// 0 to 2.
    pub temperature: Option<f32>,
    /// 0 to 1.
    pub top_p: Option<f32>,
    /// -2 to 2.
    pub presence_penalty: Option<f32>,
    /// -2 to 2.
    pub frequency_penalty: Option<f32>,
    /// Token id to bias, -100 to 100.
    pub logit_bias: Vec<(u32, f32)>,
    /// For reproducible sampling.
    pub seed: Option<u64>,
    /// Generation stops before emitting any of these. At most four.
    pub stop: Vec<String>,
}

/// Which log probabilities to return.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Logprobs {
    /// How many alternatives to give for each token, 0 to 20.
    pub top: u8,
}

/// How hard a reasoning model thinks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    None,
    Minimal,
    Low,
    Medium,
    High,
}

/// How much a model says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verbosity {
    Low,
    Medium,
    High,
}

/// An output kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    Text,
    Audio,
}

/// The voice and encoding of a spoken reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioOutput {
    pub voice: String,
    pub format: AudioFormat,
}

/// A spoken reply, or an earlier one referred to by id.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AudioReply {
    pub id: String,
    pub data: Vec<u8>,
    pub transcript: String,
    /// Unix time after which `id` can no longer be referred to.
    pub expires_at: Option<u64>,
}
