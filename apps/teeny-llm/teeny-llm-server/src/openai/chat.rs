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

//! `/v1/chat/completions`: the wire types, and their mapping to and from
//! `teeny_llm_core::chat`.
//!
//! Requests map onto [`chat::ChatRequest`] whole: what a model cannot do is refused by
//! the core's validation, naming the parameter, rather than here. Fields that concern only
//! the service (`store`, `service_tier`, `prompt_cache_key`, ...) are accepted and ignored.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use teeny_llm_core::chat;

use super::base64;
use super::error::ApiError;
use super::{FinishReason, OneOrMany, StreamOptions, Usage};

/// Message roles, as in `ChatCompletionRequestMessage` and `ChatCompletionStreamResponseDelta`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Developer,
    System,
    User,
    Assistant,
    Tool,
}

/// `image_url` of an image content part.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// `input_audio` of an audio content part: base64 data and its format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputAudio {
    pub data: String,
    pub format: String,
}

/// `file` of a file content part.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FilePart {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    /// Base64 data, optionally as a `data:` URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_data: Option<String>,
}

/// One element of an array-form message `content`, any part type (flattened).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_url: Option<ImageUrl>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_audio: Option<InputAudio>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<FilePart>,
    /// In assistant messages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

/// Message `content`: a string or an array of content parts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// The `function` of a function tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// JSON, as the model wrote it.
    pub arguments: String,
}

/// The `custom` of a custom tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CustomCall {
    pub name: String,
    pub input: String,
}

/// `ChatCompletionMessageToolCall`: a function or custom tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom: Option<CustomCall>,
}

/// A reference to an earlier spoken reply.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioRef {
    pub id: String,
}

/// `ChatCompletionRequestMessage` (all role variants, flattened).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatCompletionRequestMessage {
    pub role: Role,
    /// Nullable for assistant messages that only carry tool calls.
    #[serde(default)]
    pub content: Option<MessageContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Assistant messages: the tools it called.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Tool messages: the call answered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioRef>,
}

/// `audio` request parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioParams {
    pub voice: String,
    pub format: String,
}

/// `prediction`: predicted output content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Prediction {
    #[serde(rename = "type")]
    pub kind: String,
    pub content: MessageContent,
}

/// `CreateChatCompletionRequest`. `tools`, `tool_choice` and `response_format` are kept
/// as JSON and checked when mapped, so a malformed one is reported by its parameter path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreateChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatCompletionRequestMessage>,
    /// Deprecated in favour of `max_completion_tokens`, but still widely sent.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub stop: Option<OneOrMany>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// Token id (as a string) to bias.
    #[serde(default)]
    pub logit_bias: Option<BTreeMap<String, f32>>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<u8>,
    #[serde(default)]
    pub tools: Option<Vec<Value>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub response_format: Option<Value>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
    #[serde(default)]
    pub verbosity: Option<String>,
    #[serde(default)]
    pub modalities: Option<Vec<String>>,
    #[serde(default)]
    pub audio: Option<AudioParams>,
    #[serde(default)]
    pub prediction: Option<Prediction>,
    #[serde(default)]
    pub metadata: Option<BTreeMap<String, String>>,
    #[serde(default)]
    pub user: Option<String>,
}

fn invalid(message: impl Into<String>, param: &str) -> ApiError {
    ApiError::invalid_request(message, Some(param))
}

/// A field that must be a string, as JSON.
fn str_field<'v>(v: &'v Value, key: &str, param: &str) -> Result<&'v str, ApiError> {
    v.get(key).and_then(Value::as_str).ok_or_else(|| {
        invalid(
            format!("`{param}.{key}` must be a string"),
            &format!("{param}.{key}"),
        )
    })
}

fn content_text(content: Option<MessageContent>, param: &str) -> Result<String, ApiError> {
    match content {
        None => Ok(String::new()),
        Some(MessageContent::Text(t)) => Ok(t),
        Some(MessageContent::Parts(parts)) => parts
            .into_iter()
            .enumerate()
            .map(|(p, part)| match (part.kind.as_str(), part.text) {
                ("text", Some(t)) => Ok(t),
                ("text", None) => Err(invalid(
                    "text content part is missing `text`",
                    &format!("{param}[{p}].text"),
                )),
                (kind, _) => Err(invalid(
                    format!("a `{kind}` part is not allowed here; only `text` is"),
                    &format!("{param}[{p}].type"),
                )),
            })
            .collect(),
    }
}

fn user_parts(
    content: Option<MessageContent>,
    param: &str,
) -> Result<Vec<chat::ContentPart>, ApiError> {
    let parts = match content {
        None => return Err(invalid("a user message needs `content`", param)),
        Some(MessageContent::Text(t)) => return Ok(vec![chat::ContentPart::Text(t)]),
        Some(MessageContent::Parts(parts)) => parts,
    };
    parts
        .into_iter()
        .enumerate()
        .map(|(p, part)| {
            let at = format!("{param}[{p}]");
            Ok(match part.kind.as_str() {
                "text" => chat::ContentPart::Text(
                    part.text.ok_or_else(|| invalid("text content part is missing `text`", &format!("{at}.text")))?,
                ),
                "image_url" => {
                    let image = part
                        .image_url
                        .ok_or_else(|| invalid("image part is missing `image_url`", &format!("{at}.image_url")))?;
                    let detail = match image.detail.as_deref() {
                        None | Some("auto") => chat::ImageDetail::Auto,
                        Some("low") => chat::ImageDetail::Low,
                        Some("high") => chat::ImageDetail::High,
                        Some(d) => {
                            return Err(invalid(
                                format!("image detail `{d}` is not one of auto, low, high"),
                                &format!("{at}.image_url.detail"),
                            ));
                        }
                    };
                    chat::ContentPart::Image {
                        url: image.url,
                        detail,
                    }
                }
                "input_audio" => {
                    let audio = part
                        .input_audio
                        .ok_or_else(|| invalid("audio part is missing `input_audio`", &format!("{at}.input_audio")))?;
                    chat::ContentPart::Audio {
                        data: base64::decode(&audio.data)
                            .map_err(|e| invalid(e, &format!("{at}.input_audio.data")))?,
                        format: audio_format(&audio.format, &format!("{at}.input_audio.format"))?,
                    }
                }
                "file" => {
                    let file = part
                        .file
                        .ok_or_else(|| invalid("file part is missing `file`", &format!("{at}.file")))?;
                    let data = file
                        .file_data
                        .map(|d| {
                            let b64 = d.split_once(";base64,").map_or(d.as_str(), |(_, b)| b);
                            base64::decode(b64).map_err(|e| invalid(e, &format!("{at}.file.file_data")))
                        })
                        .transpose()?;
                    chat::ContentPart::File {
                        file_id: file.file_id,
                        filename: file.filename,
                        data,
                    }
                }
                kind => {
                    return Err(invalid(
                        format!("content part type `{kind}` is not one of text, image_url, input_audio, file"),
                        &format!("{at}.type"),
                    ));
                }
            })
        })
        .collect()
}

fn audio_format(format: &str, param: &str) -> Result<chat::AudioFormat, ApiError> {
    Ok(match format {
        "wav" => chat::AudioFormat::Wav,
        "mp3" => chat::AudioFormat::Mp3,
        "flac" => chat::AudioFormat::Flac,
        "opus" => chat::AudioFormat::Opus,
        "aac" => chat::AudioFormat::Aac,
        "pcm16" => chat::AudioFormat::Pcm16,
        f => {
            return Err(invalid(
                format!("audio format `{f}` is not one of wav, mp3, flac, opus, aac, pcm16"),
                param,
            ));
        }
    })
}

fn tool_call_to_core(call: ToolCall, param: &str) -> Result<chat::ToolCall, ApiError> {
    match (call.kind.as_str(), call.function, call.custom) {
        ("function", Some(f), _) => Ok(chat::ToolCall {
            id: call.id,
            name: f.name,
            kind: chat::ToolCallKind::Function,
            input: f.arguments,
        }),
        ("custom", _, Some(c)) => Ok(chat::ToolCall {
            id: call.id,
            name: c.name,
            kind: chat::ToolCallKind::Custom,
            input: c.input,
        }),
        (kind, _, _) => Err(invalid(
            format!("a tool call of type `{kind}` needs its `{kind}` object (function or custom)"),
            param,
        )),
    }
}

fn message_to_core(i: usize, m: ChatCompletionRequestMessage) -> Result<chat::Message, ApiError> {
    let param = format!("messages[{i}].content");
    Ok(match m.role {
        Role::Developer => chat::Message::Developer {
            content: content_text(m.content, &param)?,
            name: m.name,
        },
        Role::System => chat::Message::System {
            content: content_text(m.content, &param)?,
            name: m.name,
        },
        Role::User => chat::Message::User {
            content: user_parts(m.content, &param)?,
            name: m.name,
        },
        Role::Assistant => {
            let mut refusal = m.refusal;
            let content = match m.content {
                None => None,
                Some(MessageContent::Text(t)) => Some(t),
                Some(MessageContent::Parts(parts)) => {
                    let mut text = String::new();
                    for (p, part) in parts.into_iter().enumerate() {
                        match (part.kind.as_str(), part.text, part.refusal) {
                            ("text", Some(t), _) => text.push_str(&t),
                            ("refusal", _, Some(r)) => refusal = Some(r),
                            (kind, _, _) => {
                                return Err(invalid(
                                    format!(
                                        "an assistant message cannot hold a `{kind}` part; only text and refusal"
                                    ),
                                    &format!("{param}[{p}]"),
                                ));
                            }
                        }
                    }
                    Some(text)
                }
            };
            let tool_calls = m
                .tool_calls
                .unwrap_or_default()
                .into_iter()
                .enumerate()
                .map(|(c, call)| tool_call_to_core(call, &format!("messages[{i}].tool_calls[{c}]")))
                .collect::<Result<_, _>>()?;
            chat::Message::Assistant(chat::AssistantMessage {
                content,
                refusal,
                tool_calls,
                audio: m.audio.map(|a| chat::AudioReply {
                    id: a.id,
                    ..chat::AudioReply::default()
                }),
                name: m.name,
            })
        }
        Role::Tool => chat::Message::Tool {
            tool_call_id: m.tool_call_id.ok_or_else(|| {
                invalid(
                    "a tool message needs the `tool_call_id` of the call it answers",
                    &format!("messages[{i}].tool_call_id"),
                )
            })?,
            content: content_text(m.content, &param)?,
        },
    })
}

fn tool_to_core(i: usize, t: &Value) -> Result<chat::Tool, ApiError> {
    let param = format!("tools[{i}]");
    let kind = str_field(t, "type", &param)?;
    let body = t.get(kind).ok_or_else(|| {
        invalid(
            format!("a `{kind}` tool needs its `{kind}` object"),
            &format!("{param}.{kind}"),
        )
    })?;
    let at = format!("{param}.{kind}");
    let name = str_field(body, "name", &at)?.to_owned();
    let description = body
        .get("description")
        .and_then(Value::as_str)
        .map(str::to_owned);
    let kind = match kind {
        "function" => chat::ToolKind::Function {
            parameters: body.get("parameters").filter(|p| !p.is_null()).cloned(),
            strict: body.get("strict").and_then(Value::as_bool).unwrap_or(false),
        },
        "custom" => {
            let format = match body.get("format") {
                None | Some(Value::Null) => chat::CustomFormat::Text,
                Some(f) => match f.get("type").and_then(Value::as_str) {
                    Some("text") => chat::CustomFormat::Text,
                    Some("grammar") => {
                        let g = f.get("grammar").ok_or_else(|| {
                            invalid(
                                "a grammar format needs `grammar`",
                                &format!("{at}.format.grammar"),
                            )
                        })?;
                        let syntax = match str_field(g, "syntax", &format!("{at}.format.grammar"))?
                        {
                            "lark" => chat::GrammarSyntax::Lark,
                            "regex" => chat::GrammarSyntax::Regex,
                            s => {
                                return Err(invalid(
                                    format!("grammar syntax `{s}` is not lark or regex"),
                                    &format!("{at}.format.grammar.syntax"),
                                ));
                            }
                        };
                        chat::CustomFormat::Grammar {
                            syntax,
                            definition: str_field(
                                g,
                                "definition",
                                &format!("{at}.format.grammar"),
                            )?
                            .to_owned(),
                        }
                    }
                    _ => {
                        return Err(invalid(
                            "a custom tool format's `type` must be text or grammar",
                            &format!("{at}.format.type"),
                        ));
                    }
                },
            };
            chat::ToolKind::Custom { format }
        }
        k => {
            return Err(invalid(
                format!("tool type `{k}` is not function or custom"),
                &format!("{param}.type"),
            ));
        }
    };
    Ok(chat::Tool {
        name,
        description,
        kind,
    })
}

/// The name a `{"type": "function", "function": {"name": ...}}` (or custom) tool
/// reference names.
fn named_tool(v: &Value, param: &str) -> Result<String, ApiError> {
    let kind = str_field(v, "type", param)?;
    let body = v.get(kind).ok_or_else(|| {
        invalid(
            format!("needs its `{kind}` object"),
            &format!("{param}.{kind}"),
        )
    })?;
    Ok(str_field(body, "name", &format!("{param}.{kind}"))?.to_owned())
}

fn tool_choice_to_core(v: &Value) -> Result<chat::ToolChoice, ApiError> {
    const P: &str = "tool_choice";
    Ok(match v {
        Value::String(s) => match s.as_str() {
            "auto" => chat::ToolChoice::Auto,
            "none" => chat::ToolChoice::None,
            "required" => chat::ToolChoice::Required,
            s => return Err(invalid(format!("`{s}` is not auto, none or required"), P)),
        },
        Value::Object(_) => match str_field(v, "type", P)? {
            "function" | "custom" => chat::ToolChoice::Named(named_tool(v, P)?),
            "allowed_tools" => {
                let a = v.get("allowed_tools").ok_or_else(|| {
                    invalid(
                        "needs its `allowed_tools` object",
                        "tool_choice.allowed_tools",
                    )
                })?;
                let mode = match str_field(a, "mode", "tool_choice.allowed_tools")? {
                    "auto" => chat::AllowedMode::Auto,
                    "required" => chat::AllowedMode::Required,
                    m => {
                        return Err(invalid(
                            format!("mode `{m}` is not auto or required"),
                            "tool_choice.allowed_tools.mode",
                        ));
                    }
                };
                let tools = a
                    .get("tools")
                    .and_then(Value::as_array)
                    .ok_or_else(|| invalid("must be an array", "tool_choice.allowed_tools.tools"))?
                    .iter()
                    .enumerate()
                    .map(|(i, t)| named_tool(t, &format!("tool_choice.allowed_tools.tools[{i}]")))
                    .collect::<Result<_, _>>()?;
                chat::ToolChoice::Allowed { mode, tools }
            }
            t => {
                return Err(invalid(
                    format!("type `{t}` is not function, custom or allowed_tools"),
                    "tool_choice.type",
                ));
            }
        },
        _ => return Err(invalid("must be a string or an object", P)),
    })
}

fn response_format_to_core(v: &Value) -> Result<chat::ResponseFormat, ApiError> {
    const P: &str = "response_format";
    Ok(match str_field(v, "type", P)? {
        "text" => chat::ResponseFormat::Text,
        "json_object" => chat::ResponseFormat::JsonObject,
        "json_schema" => {
            let s = v.get("json_schema").ok_or_else(|| {
                invalid(
                    "needs its `json_schema` object",
                    "response_format.json_schema",
                )
            })?;
            chat::ResponseFormat::JsonSchema {
                name: str_field(s, "name", "response_format.json_schema")?.to_owned(),
                description: s
                    .get("description")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                schema: s
                    .get("schema")
                    .cloned()
                    .unwrap_or(Value::Object(Default::default())),
                strict: s.get("strict").and_then(Value::as_bool).unwrap_or(false),
            }
        }
        t => {
            return Err(invalid(
                format!("type `{t}` is not text, json_object or json_schema"),
                "response_format.type",
            ));
        }
    })
}

impl CreateChatCompletionRequest {
    /// Whether the reply is to be streamed.
    pub fn streamed(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    /// The request as the core's [`chat::ChatRequest`].
    pub fn into_core(self) -> Result<chat::ChatRequest, ApiError> {
        let messages = self
            .messages
            .into_iter()
            .enumerate()
            .map(|(i, m)| message_to_core(i, m))
            .collect::<Result<_, _>>()?;
        let tools = self
            .tools
            .unwrap_or_default()
            .iter()
            .enumerate()
            .map(|(i, t)| tool_to_core(i, t))
            .collect::<Result<_, _>>()?;
        let logit_bias = self
            .logit_bias
            .unwrap_or_default()
            .into_iter()
            .map(|(token, bias)| {
                token.parse::<u32>().map(|id| (id, bias)).map_err(|_| {
                    invalid(
                        format!("`{token}` is not a token id"),
                        &format!("logit_bias.{token}"),
                    )
                })
            })
            .collect::<Result<_, _>>()?;
        let logprobs = match (self.logprobs, self.top_logprobs) {
            (Some(true), top) => Some(chat::Logprobs {
                top: top.unwrap_or(0),
            }),
            (_, Some(_)) => {
                return Err(invalid(
                    "`top_logprobs` needs `logprobs: true`",
                    "top_logprobs",
                ));
            }
            _ => None,
        };
        let reasoning_effort = self
            .reasoning_effort
            .as_deref()
            .map(|e| {
                Ok(match e {
                    "none" => chat::ReasoningEffort::None,
                    "minimal" => chat::ReasoningEffort::Minimal,
                    "low" => chat::ReasoningEffort::Low,
                    "medium" => chat::ReasoningEffort::Medium,
                    "high" => chat::ReasoningEffort::High,
                    e => {
                        return Err(invalid(
                            format!("`{e}` is not none, minimal, low, medium or high"),
                            "reasoning_effort",
                        ));
                    }
                })
            })
            .transpose()?;
        let verbosity = self
            .verbosity
            .as_deref()
            .map(|v| {
                Ok(match v {
                    "low" => chat::Verbosity::Low,
                    "medium" => chat::Verbosity::Medium,
                    "high" => chat::Verbosity::High,
                    v => {
                        return Err(invalid(
                            format!("`{v}` is not low, medium or high"),
                            "verbosity",
                        ));
                    }
                })
            })
            .transpose()?;
        let modalities = self
            .modalities
            .unwrap_or_default()
            .iter()
            .enumerate()
            .map(|(i, m)| match m.as_str() {
                "text" => Ok(chat::Modality::Text),
                "audio" => Ok(chat::Modality::Audio),
                m => Err(invalid(
                    format!("`{m}` is not text or audio"),
                    &format!("modalities[{i}]"),
                )),
            })
            .collect::<Result<_, _>>()?;
        let audio = self
            .audio
            .map(|a| {
                Ok::<_, ApiError>(chat::AudioOutput {
                    voice: a.voice,
                    format: audio_format(&a.format, "audio.format")?,
                })
            })
            .transpose()?;
        let prediction = self
            .prediction
            .map(|p| content_text(Some(p.content), "prediction.content"))
            .transpose()?;
        Ok(chat::ChatRequest {
            model: self.model,
            messages,
            tools,
            tool_choice: self
                .tool_choice
                .as_ref()
                .map(tool_choice_to_core)
                .transpose()?
                .unwrap_or_default(),
            parallel_tool_calls: self.parallel_tool_calls,
            response_format: self
                .response_format
                .as_ref()
                .map(response_format_to_core)
                .transpose()?
                .unwrap_or_default(),
            sampling: chat::Sampling {
                temperature: self.temperature,
                top_p: self.top_p,
                presence_penalty: self.presence_penalty,
                frequency_penalty: self.frequency_penalty,
                logit_bias,
                seed: self.seed.map(i64::cast_unsigned),
                stop: self.stop.map(OneOrMany::into_vec).unwrap_or_default(),
            },
            max_completion_tokens: self.max_completion_tokens.or(self.max_tokens),
            n: self.n.unwrap_or(1),
            logprobs,
            reasoning_effort,
            verbosity,
            modalities,
            audio,
            prediction,
            include_usage: self
                .stream_options
                .and_then(|o| o.include_usage)
                .unwrap_or(false),
            user: self.user,
            metadata: self.metadata.unwrap_or_default().into_iter().collect(),
        })
    }
}

/// A spoken reply.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioResponse {
    pub id: String,
    /// Base64.
    pub data: String,
    pub transcript: String,
    pub expires_at: u64,
}

/// `ChatCompletionResponseMessage`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatCompletionResponseMessage {
    pub role: Role,
    /// Required-but-nullable.
    pub content: Option<String>,
    pub refusal: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioResponse>,
}

/// An alternative token at a position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
}

/// `ChatCompletionTokenLogprob`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<TopLogprob>,
}

/// A choice's `logprobs`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChoiceLogprobs {
    pub content: Option<Vec<TokenLogprob>>,
    pub refusal: Option<Vec<TokenLogprob>>,
}

/// One element of `CreateChatCompletionResponse.choices`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatCompletionResponseMessage,
    /// Required-but-nullable: `null` unless `logprobs` was asked for.
    pub logprobs: Option<ChoiceLogprobs>,
    pub finish_reason: FinishReason,
}

/// `CreateChatCompletionResponse`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreateChatCompletionResponse {
    pub id: String,
    /// Always `"chat.completion"`.
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// The `function` of a streamed tool call: the name on its first chunk, then pieces of
/// the arguments.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionCallChunk {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

/// The `custom` of a streamed custom tool call.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CustomCallChunk {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
}

/// `ChatCompletionMessageToolCallChunk`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallChunk {
    pub index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub function: Option<FunctionCallChunk>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom: Option<CustomCallChunk>,
}

/// `ChatCompletionStreamResponseDelta`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatCompletionStreamResponseDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<Role>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallChunk>>,
}

/// One element of `CreateChatCompletionStreamResponse.choices`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionStreamChoice {
    pub index: u32,
    pub delta: ChatCompletionStreamResponseDelta,
    pub logprobs: Option<ChoiceLogprobs>,
    pub finish_reason: Option<FinishReason>,
}

/// `CreateChatCompletionStreamResponse`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreateChatCompletionStreamResponse {
    pub id: String,
    /// Always `"chat.completion.chunk"`.
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionStreamChoice>,
    /// Only on the final chunk, and only with `stream_options.include_usage`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

impl From<chat::ChatFinishReason> for FinishReason {
    fn from(r: chat::ChatFinishReason) -> Self {
        match r {
            chat::ChatFinishReason::Stop => Self::Stop,
            chat::ChatFinishReason::Length => Self::Length,
            chat::ChatFinishReason::ToolCalls => Self::ToolCalls,
            chat::ChatFinishReason::ContentFilter => Self::ContentFilter,
        }
    }
}

fn logprobs_from_core(v: Vec<chat::TokenLogprob>) -> Vec<TokenLogprob> {
    v.into_iter()
        .map(|t| TokenLogprob {
            token: t.token,
            logprob: t.logprob,
            bytes: Some(t.bytes),
            top_logprobs: t
                .top
                .into_iter()
                .map(|a| TopLogprob {
                    token: a.token,
                    logprob: a.logprob,
                    bytes: Some(a.bytes),
                })
                .collect(),
        })
        .collect()
}

/// A choice's `logprobs` when they were asked for, else `None`.
pub fn choice_logprobs(
    asked: bool,
    content: Vec<chat::TokenLogprob>,
    refusal: Vec<chat::TokenLogprob>,
) -> Option<ChoiceLogprobs> {
    asked.then(|| ChoiceLogprobs {
        content: Some(logprobs_from_core(content)),
        refusal: (!refusal.is_empty()).then(|| logprobs_from_core(refusal)),
    })
}

/// A tool call as the wire writes it.
pub fn tool_call_from_core(c: chat::ToolCall) -> ToolCall {
    match c.kind {
        chat::ToolCallKind::Function => ToolCall {
            id: c.id,
            kind: "function".into(),
            function: Some(FunctionCall {
                name: c.name,
                arguments: c.input,
            }),
            custom: None,
        },
        chat::ToolCallKind::Custom => ToolCall {
            id: c.id,
            kind: "custom".into(),
            function: None,
            custom: Some(CustomCall {
                name: c.name,
                input: c.input,
            }),
        },
    }
}

/// A finished choice as the wire writes it.
pub fn choice_from_core(c: chat::Choice, logprobs: bool) -> ChatCompletionChoice {
    let m = c.message;
    ChatCompletionChoice {
        index: c.index,
        message: ChatCompletionResponseMessage {
            role: Role::Assistant,
            content: m.content,
            refusal: m.refusal,
            tool_calls: (!m.tool_calls.is_empty())
                .then(|| m.tool_calls.into_iter().map(tool_call_from_core).collect()),
            audio: m.audio.map(|a| AudioResponse {
                id: a.id,
                data: base64::encode(&a.data),
                transcript: a.transcript,
                expires_at: a.expires_at.unwrap_or(0),
            }),
        },
        logprobs: choice_logprobs(logprobs, c.content_logprobs, c.refusal_logprobs),
        finish_reason: c.finish_reason.into(),
    }
}

/// A streamed delta as the wire writes it.
pub fn delta_from_core(d: chat::Delta) -> ChatCompletionStreamResponseDelta {
    ChatCompletionStreamResponseDelta {
        role: None,
        content: d.content,
        refusal: d.refusal,
        tool_calls: (!d.tool_calls.is_empty()).then(|| {
            d.tool_calls
                .into_iter()
                .map(|t| {
                    let kind = t.kind;
                    let custom = matches!(kind, Some(chat::ToolCallKind::Custom));
                    let input = Some(t.input).filter(|s| !s.is_empty());
                    ToolCallChunk {
                        index: t.index,
                        id: t.id,
                        kind: kind.map(|k| match k {
                            chat::ToolCallKind::Function => "function".into(),
                            chat::ToolCallKind::Custom => "custom".into(),
                        }),
                        function: (!custom).then(|| FunctionCallChunk {
                            name: t.name.clone(),
                            arguments: input.clone(),
                        }),
                        custom: custom.then_some(CustomCallChunk {
                            name: t.name,
                            input,
                        }),
                    }
                })
                .collect()
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(v: Value) -> Result<chat::ChatRequest, ApiError> {
        serde_json::from_value::<CreateChatCompletionRequest>(v)
            .unwrap()
            .into_core()
    }

    #[test]
    fn test_maps_a_full_tool_conversation() {
        let r = parse(json!({
            "model": "m",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": [
                    {"type": "text", "text": "what is "},
                    {"type": "image_url", "image_url": {"url": "https://x/y.png", "detail": "low"}}
                ]},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\":\"Oslo\"}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "{\"c\":3}"}
            ],
            "tools": [{"type": "function", "function": {
                "name": "get_weather", "description": "d",
                "parameters": {"type": "object"}, "strict": true}}],
            "tool_choice": {"type": "allowed_tools", "allowed_tools": {
                "mode": "required", "tools": [{"type": "function", "function": {"name": "get_weather"}}]}},
            "parallel_tool_calls": false,
            "response_format": {"type": "json_schema", "json_schema": {"name": "w", "schema": {}}},
            "n": 2, "logprobs": true, "top_logprobs": 3,
            "logit_bias": {"50256": -100},
            "stream_options": {"include_usage": true},
            "store": true, "service_tier": "auto"
        }))
        .unwrap();
        assert_eq!(r.messages.len(), 4);
        assert!(matches!(&r.messages[1], chat::Message::User { content, .. }
            if matches!(content[1], chat::ContentPart::Image { detail: chat::ImageDetail::Low, .. })));
        let chat::Message::Assistant(a) = &r.messages[2] else {
            panic!("not an assistant message");
        };
        assert_eq!(a.tool_calls[0].input, r#"{"city":"Oslo"}"#);
        assert!(
            matches!(&r.messages[3], chat::Message::Tool { tool_call_id, .. } if tool_call_id == "call_1")
        );
        assert!(matches!(
            r.tools[0].kind,
            chat::ToolKind::Function { strict: true, .. }
        ));
        assert_eq!(
            r.tool_choice,
            chat::ToolChoice::Allowed {
                mode: chat::AllowedMode::Required,
                tools: vec!["get_weather".into()],
            }
        );
        assert_eq!(r.parallel_tool_calls, Some(false));
        assert!(matches!(
            r.response_format,
            chat::ResponseFormat::JsonSchema { .. }
        ));
        assert_eq!((r.n, r.logprobs), (2, Some(chat::Logprobs { top: 3 })));
        assert_eq!(r.sampling.logit_bias, vec![(50256, -100.0)]);
        assert!(r.include_usage);
    }

    #[test]
    fn test_reports_malformed_fields_by_path() {
        let cases = [
            (json!({"tool_choice": "sometimes"}), "tool_choice"),
            (
                json!({"tools": [{"type": "function", "function": {}}]}),
                "tools[0].function.name",
            ),
            (
                json!({"response_format": {"type": "yaml"}}),
                "response_format.type",
            ),
            (json!({"logit_bias": {"x": 1}}), "logit_bias.x"),
            (json!({"top_logprobs": 2}), "top_logprobs"),
        ];
        for (extra, param) in cases {
            let mut v = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
            v.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let err = parse(v).unwrap_err();
            assert_eq!(
                err.body.param.as_deref(),
                Some(param),
                "{}",
                err.body.message
            );
        }
        let err = parse(json!({"model": "m", "messages": [{"role": "tool", "content": "x"}]}))
            .unwrap_err();
        assert_eq!(err.body.param.as_deref(), Some("messages[0].tool_call_id"));
    }

    #[test]
    fn test_streamed_tool_call_chunks_follow_the_spec() {
        let first = delta_from_core(chat::Delta {
            tool_calls: vec![chat::ToolCallDelta {
                index: 0,
                id: Some("call_0".into()),
                name: Some("f".into()),
                kind: Some(chat::ToolCallKind::Function),
                input: String::new(),
            }],
            ..chat::Delta::default()
        });
        assert_eq!(
            serde_json::to_value(&first).unwrap(),
            json!({"tool_calls": [{"index": 0, "id": "call_0", "type": "function",
                                   "function": {"name": "f"}}]})
        );
        let more = delta_from_core(chat::Delta {
            tool_calls: vec![chat::ToolCallDelta {
                index: 0,
                input: "{\"a\":".into(),
                ..chat::ToolCallDelta::default()
            }],
            ..chat::Delta::default()
        });
        assert_eq!(
            serde_json::to_value(&more).unwrap(),
            json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"a\":"}}]})
        );
    }
}
