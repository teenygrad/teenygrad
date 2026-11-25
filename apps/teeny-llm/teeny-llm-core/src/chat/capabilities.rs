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

//! What a chat model can do, and checking a request against it.
//!
//! [`validate`] runs before a request reaches a model, so a model only ever sees requests
//! it declared it can serve, and a client asking for something a model lacks gets an error
//! naming the parameter instead of a reply that quietly ignored it.

use std::collections::HashSet;

use super::request::{
    AllowedMode, ChatRequest, ContentPart, Message, Modality, ResponseFormat, ToolChoice, ToolKind,
};
use crate::{Error, Result};

/// The OpenAI limit on stop strings.
pub const MAX_STOP_STRINGS: usize = 4;

/// The OpenAI limit on `top_logprobs`.
pub const MAX_TOP_LOGPROBS: u8 = 20;

/// Everything a request can ask for that a model might not offer.
///
/// `Default` is the least a chat model can be: text in and out, greedy decoding, one
/// choice, no tools. Models switch on what they support.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capabilities {
    /// Most tokens of prompt and reply together.
    pub context_length: u32,
    /// Most tokens in one reply, when lower than what the context leaves.
    pub max_output_tokens: Option<u32>,
    pub tools: ToolSupport,
    pub structured_output: StructuredOutput,
    pub input: InputModalities,
    /// Whether the model can reply in speech.
    pub audio_output: bool,
    pub sampling: SamplingSupport,
    /// Whether generation can stop at caller-given strings.
    pub stop_strings: bool,
    /// The most `top_logprobs` the model can report, or `None` without log probabilities.
    pub max_top_logprobs: Option<u8>,
    /// The most choices (`n`) one request can ask for.
    pub max_choices: u32,
    /// Whether `reasoning_effort` means anything to the model.
    pub reasoning: bool,
    /// Whether `verbosity` means anything to the model.
    pub verbosity: bool,
    /// Whether the model can use predicted outputs.
    pub predicted_outputs: bool,
}

impl Default for Capabilities {
    fn default() -> Self {
        Self {
            context_length: 2048,
            max_output_tokens: None,
            tools: ToolSupport::default(),
            structured_output: StructuredOutput::default(),
            input: InputModalities::default(),
            audio_output: false,
            sampling: SamplingSupport::default(),
            stop_strings: false,
            max_top_logprobs: None,
            max_choices: 1,
            reasoning: false,
            verbosity: false,
            predicted_outputs: false,
        }
    }
}

/// Which tool calling a model does.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ToolSupport {
    /// Function tools, with JSON arguments.
    pub functions: bool,
    /// Custom tools, with free-text input.
    pub custom: bool,
    /// Whether one turn can make more than one call.
    pub parallel: bool,
    /// Whether the model can be made to call (`required`, a named tool, `allowed_tools`),
    /// as opposed to only deciding for itself.
    pub forced: bool,
    /// Whether `strict` function arguments (exactly matching the schema) are enforced.
    pub strict: bool,
}

/// Which constrained reply formats a model can produce.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StructuredOutput {
    pub json_object: bool,
    pub json_schema: bool,
}

/// Which kinds of user content a model can read, beyond text.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InputModalities {
    pub image: bool,
    pub audio: bool,
    pub file: bool,
}

/// Which decoding controls a model honours. All off means greedy decoding.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SamplingSupport {
    pub temperature: bool,
    pub top_p: bool,
    /// `presence_penalty` and `frequency_penalty`.
    pub penalties: bool,
    pub logit_bias: bool,
    pub seed: bool,
}

fn invalid(parameter: impl Into<String>, detail: impl Into<String>) -> Error {
    Error::InvalidParameter {
        parameter: parameter.into(),
        detail: detail.into(),
    }
}

/// Checks `request` for the model `model` with capabilities `caps`: first that it is
/// well formed whatever the model (ranges, references between messages and tools), then
/// that the model can serve every part of it. The first problem found is returned, naming
/// the parameter (`messages[2].content[0]`, `tool_choice`, ...) the way OpenAI's `param`
/// does.
pub fn validate(model: &str, caps: &Capabilities, request: &ChatRequest) -> Result<()> {
    check_well_formed(request)?;
    check_supported(model, caps, request)
}

fn check_range(parameter: &str, value: Option<f32>, lo: f32, hi: f32) -> Result<()> {
    match value {
        Some(v) if !(lo..=hi).contains(&v) => {
            Err(invalid(parameter, format!("{v} is outside {lo} to {hi}")))
        }
        _ => Ok(()),
    }
}

fn check_well_formed(r: &ChatRequest) -> Result<()> {
    if r.messages.is_empty() {
        return Err(invalid("messages", "a chat needs at least one message"));
    }
    if r.n == 0 {
        return Err(invalid("n", "at least one choice must be asked for"));
    }
    if r.max_completion_tokens == Some(0) {
        return Err(invalid(
            "max_completion_tokens",
            "must be at least 1 when given",
        ));
    }
    let s = &r.sampling;
    check_range("temperature", s.temperature, 0.0, 2.0)?;
    check_range("top_p", s.top_p, 0.0, 1.0)?;
    check_range("presence_penalty", s.presence_penalty, -2.0, 2.0)?;
    check_range("frequency_penalty", s.frequency_penalty, -2.0, 2.0)?;
    for (token, bias) in &s.logit_bias {
        check_range(&format!("logit_bias.{token}"), Some(*bias), -100.0, 100.0)?;
    }
    if s.stop.len() > MAX_STOP_STRINGS {
        return Err(invalid(
            "stop",
            format!(
                "{} stop strings given, at most {MAX_STOP_STRINGS} are allowed",
                s.stop.len()
            ),
        ));
    }
    if let Some(i) = s.stop.iter().position(String::is_empty) {
        return Err(invalid(
            format!("stop[{i}]"),
            "a stop string cannot be empty",
        ));
    }
    if let Some(l) = r.logprobs
        && l.top > MAX_TOP_LOGPROBS
    {
        return Err(invalid(
            "top_logprobs",
            format!("{} is above the limit of {MAX_TOP_LOGPROBS}", l.top),
        ));
    }
    if r.modalities.contains(&Modality::Audio) != r.audio.is_some() {
        return Err(invalid(
            "audio",
            "audio output needs both `modalities` to include \"audio\" and the `audio` options",
        ));
    }

    let mut names = HashSet::new();
    for (i, t) in r.tools.iter().enumerate() {
        if t.name.is_empty() {
            return Err(invalid(format!("tools[{i}].name"), "a tool needs a name"));
        }
        if !names.insert(t.name.as_str()) {
            return Err(invalid(
                format!("tools[{i}].name"),
                format!("`{}` is declared more than once", t.name),
            ));
        }
    }
    let declared = |name: &str, parameter: &str| -> Result<()> {
        if names.contains(name) {
            Ok(())
        } else {
            Err(invalid(
                parameter,
                format!("names `{name}`, which is not among the declared tools"),
            ))
        }
    };
    match &r.tool_choice {
        ToolChoice::Required if r.tools.is_empty() => {
            return Err(invalid(
                "tool_choice",
                "`required` needs at least one tool in `tools`",
            ));
        }
        ToolChoice::Named(name) => declared(name, "tool_choice")?,
        ToolChoice::Allowed { tools, .. } => {
            if tools.is_empty() {
                return Err(invalid(
                    "tool_choice.allowed_tools",
                    "list at least one tool",
                ));
            }
            for (i, name) in tools.iter().enumerate() {
                declared(name, &format!("tool_choice.allowed_tools[{i}]"))?;
            }
        }
        _ => {}
    }

    // Every tool result must answer a call made earlier in the conversation, and each call
    // is answered at most once.
    let mut open: HashSet<&str> = HashSet::new();
    let mut answered: HashSet<&str> = HashSet::new();
    for (i, m) in r.messages.iter().enumerate() {
        match m {
            Message::Assistant(a) => {
                if a.content.is_none() && a.refusal.is_none() && a.tool_calls.is_empty() {
                    return Err(invalid(
                        format!("messages[{i}]"),
                        "an assistant message needs content, a refusal or tool calls",
                    ));
                }
                for (j, c) in a.tool_calls.iter().enumerate() {
                    if c.id.is_empty() {
                        return Err(invalid(
                            format!("messages[{i}].tool_calls[{j}].id"),
                            "a tool call needs an id",
                        ));
                    }
                    open.insert(c.id.as_str());
                }
            }
            Message::Tool { tool_call_id, .. } => {
                let id = tool_call_id.as_str();
                if !open.contains(id) {
                    return Err(invalid(
                        format!("messages[{i}].tool_call_id"),
                        format!("`{id}` answers no tool call made before it"),
                    ));
                }
                if !answered.insert(id) {
                    return Err(invalid(
                        format!("messages[{i}].tool_call_id"),
                        format!("tool call `{id}` is answered more than once"),
                    ));
                }
            }
            Message::User { content, .. } if content.is_empty() => {
                return Err(invalid(
                    format!("messages[{i}].content"),
                    "a user message needs content",
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn check_supported(model: &str, caps: &Capabilities, r: &ChatRequest) -> Result<()> {
    let unsupported = |parameter: &str, detail: String| Error::UnsupportedParameter {
        model: model.to_owned(),
        parameter: parameter.to_owned(),
        detail,
    };

    if r.n > caps.max_choices {
        return Err(unsupported(
            "n",
            format!(
                "{} choices asked for, at most {} per request",
                r.n, caps.max_choices
            ),
        ));
    }
    let output_limit = caps.max_output_tokens.unwrap_or(caps.context_length);
    if let Some(max) = r.max_completion_tokens
        && max > output_limit
    {
        return Err(unsupported(
            "max_completion_tokens",
            format!("{max} is above the model's limit of {output_limit} tokens per reply"),
        ));
    }

    // Decoding controls.
    let s = &r.sampling;
    let sc = &caps.sampling;
    // Greedy decoding is what temperature 0 asks for, so a greedy model can honour it.
    if !sc.temperature && s.temperature.is_some_and(|t| t != 0.0) {
        return Err(unsupported(
            "temperature",
            "the model decodes greedily; leave it out or pass 0".into(),
        ));
    }
    if !sc.top_p && s.top_p.is_some_and(|p| p != 1.0) {
        return Err(unsupported(
            "top_p",
            "the model does not do nucleus sampling; leave it out or pass 1".into(),
        ));
    }
    for (name, v) in [
        ("presence_penalty", s.presence_penalty),
        ("frequency_penalty", s.frequency_penalty),
    ] {
        if !sc.penalties && v.is_some_and(|v| v != 0.0) {
            return Err(unsupported(
                name,
                "the model applies no repetition penalties; leave it out or pass 0".into(),
            ));
        }
    }
    if !sc.logit_bias && !s.logit_bias.is_empty() {
        return Err(unsupported(
            "logit_bias",
            "the model cannot bias token probabilities".into(),
        ));
    }
    if !sc.seed && s.seed.is_some() && sc != &SamplingSupport::default() {
        // A greedy model is reproducible anyway, so a seed asks nothing of it; a sampling
        // model that cannot be seeded would silently ignore it.
        return Err(unsupported(
            "seed",
            "the model samples but cannot be seeded".into(),
        ));
    }
    if !caps.stop_strings && !s.stop.is_empty() {
        return Err(unsupported(
            "stop",
            "the model cannot stop at caller-given strings".into(),
        ));
    }
    if let Some(l) = r.logprobs {
        match caps.max_top_logprobs {
            None => {
                return Err(unsupported(
                    "logprobs",
                    "the model does not report log probabilities".into(),
                ));
            }
            Some(max) if l.top > max => {
                return Err(unsupported(
                    "top_logprobs",
                    format!("{} asked for, the model reports at most {max}", l.top),
                ));
            }
            _ => {}
        }
    }

    // Tools.
    let t = &caps.tools;
    for (i, tool) in r.tools.iter().enumerate() {
        match &tool.kind {
            ToolKind::Function { strict, .. } => {
                if !t.functions {
                    return Err(unsupported(
                        &format!("tools[{i}]"),
                        "the model does not call functions".into(),
                    ));
                }
                if *strict && !t.strict {
                    return Err(unsupported(
                        &format!("tools[{i}].function.strict"),
                        "the model cannot guarantee arguments that match the schema".into(),
                    ));
                }
            }
            ToolKind::Custom { .. } if !t.custom => {
                return Err(unsupported(
                    &format!("tools[{i}]"),
                    "the model does not call custom (free-text) tools".into(),
                ));
            }
            ToolKind::Custom { .. } => {}
        }
    }
    let forced = match &r.tool_choice {
        ToolChoice::Required | ToolChoice::Named(_) => true,
        ToolChoice::Allowed { mode, .. } => *mode == AllowedMode::Required,
        ToolChoice::Auto | ToolChoice::None => false,
    };
    if forced && !t.forced {
        return Err(unsupported(
            "tool_choice",
            "the model decides for itself whether to call a tool, and cannot be made to".into(),
        ));
    }
    if r.parallel_tool_calls == Some(true) && !t.parallel && !r.tools.is_empty() {
        return Err(unsupported(
            "parallel_tool_calls",
            "the model makes at most one tool call per turn".into(),
        ));
    }

    // Reply format.
    match &r.response_format {
        ResponseFormat::Text => {}
        ResponseFormat::JsonObject if !caps.structured_output.json_object => {
            return Err(unsupported(
                "response_format",
                "the model cannot guarantee a JSON object reply".into(),
            ));
        }
        ResponseFormat::JsonSchema { .. } if !caps.structured_output.json_schema => {
            return Err(unsupported(
                "response_format",
                "the model cannot guarantee a reply matching a JSON schema".into(),
            ));
        }
        _ => {}
    }

    // Input and output modalities.
    for (i, m) in r.messages.iter().enumerate() {
        let Message::User { content, .. } = m else {
            continue;
        };
        for (j, part) in content.iter().enumerate() {
            let ok = match part {
                ContentPart::Text(_) => true,
                ContentPart::Image { .. } => caps.input.image,
                ContentPart::Audio { .. } => caps.input.audio,
                ContentPart::File { .. } => caps.input.file,
            };
            if !ok {
                return Err(unsupported(
                    &format!("messages[{i}].content[{j}]"),
                    format!("the model cannot read `{}` content", part.kind()),
                ));
            }
        }
    }
    if r.modalities.contains(&Modality::Audio) && !caps.audio_output {
        return Err(unsupported(
            "modalities",
            "the model cannot reply in audio".into(),
        ));
    }

    if r.reasoning_effort.is_some() && !caps.reasoning {
        return Err(unsupported(
            "reasoning_effort",
            "the model does not reason before answering".into(),
        ));
    }
    if r.verbosity.is_some() && !caps.verbosity {
        return Err(unsupported(
            "verbosity",
            "the model takes no verbosity hint".into(),
        ));
    }
    if r.prediction.is_some() && !caps.predicted_outputs {
        return Err(unsupported(
            "prediction",
            "the model cannot use predicted outputs".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::request::{
        AssistantMessage, AudioFormat, ImageDetail, Logprobs, Sampling, Tool, ToolCall,
        ToolCallKind,
    };

    fn weather() -> Tool {
        Tool {
            name: "get_weather".into(),
            description: None,
            kind: ToolKind::Function {
                parameters: None,
                strict: false,
            },
        }
    }

    fn request() -> ChatRequest {
        ChatRequest {
            model: "m".into(),
            messages: vec![Message::user("hi")],
            ..ChatRequest::default()
        }
    }

    fn tools_caps() -> Capabilities {
        Capabilities {
            tools: ToolSupport {
                functions: true,
                ..ToolSupport::default()
            },
            ..Capabilities::default()
        }
    }

    /// The parameter an error names, or a panic when the request passed.
    fn param(caps: &Capabilities, r: &ChatRequest) -> (String, bool) {
        match validate("m", caps, r) {
            Err(Error::InvalidParameter { parameter, .. }) => (parameter, false),
            Err(Error::UnsupportedParameter { parameter, .. }) => (parameter, true),
            other => panic!("expected a parameter error, got {other:?}"),
        }
    }

    #[test]
    fn test_validate_accepts_a_plain_request_for_the_least_capable_model() {
        validate("m", &Capabilities::default(), &request()).unwrap();
    }

    #[test]
    fn test_validate_rejects_malformed_requests_whatever_the_model() {
        let all = Capabilities {
            max_choices: 8,
            stop_strings: true,
            max_top_logprobs: Some(20),
            sampling: SamplingSupport {
                temperature: true,
                top_p: true,
                penalties: true,
                logit_bias: true,
                seed: true,
            },
            ..tools_caps()
        };
        let cases: Vec<(ChatRequest, &str)> = vec![
            (
                ChatRequest {
                    messages: vec![],
                    ..request()
                },
                "messages",
            ),
            (ChatRequest { n: 0, ..request() }, "n"),
            (
                ChatRequest {
                    max_completion_tokens: Some(0),
                    ..request()
                },
                "max_completion_tokens",
            ),
            (
                ChatRequest {
                    sampling: Sampling {
                        temperature: Some(2.5),
                        ..Sampling::default()
                    },
                    ..request()
                },
                "temperature",
            ),
            (
                ChatRequest {
                    sampling: Sampling {
                        logit_bias: vec![(7, 101.0)],
                        ..Sampling::default()
                    },
                    ..request()
                },
                "logit_bias.7",
            ),
            (
                ChatRequest {
                    sampling: Sampling {
                        stop: vec!["a".into(); 5],
                        ..Sampling::default()
                    },
                    ..request()
                },
                "stop",
            ),
            (
                ChatRequest {
                    logprobs: Some(Logprobs { top: 21 }),
                    ..request()
                },
                "top_logprobs",
            ),
            (
                ChatRequest {
                    tool_choice: ToolChoice::Required,
                    ..request()
                },
                "tool_choice",
            ),
            (
                ChatRequest {
                    tools: vec![weather()],
                    tool_choice: ToolChoice::Named("get_time".into()),
                    ..request()
                },
                "tool_choice",
            ),
            (
                ChatRequest {
                    tools: vec![weather(), weather()],
                    ..request()
                },
                "tools[1].name",
            ),
            (
                ChatRequest {
                    modalities: vec![Modality::Audio],
                    ..request()
                },
                "audio",
            ),
            (
                ChatRequest {
                    messages: vec![
                        Message::user("hi"),
                        Message::Tool {
                            tool_call_id: "call_1".into(),
                            content: "{}".into(),
                        },
                    ],
                    ..request()
                },
                "messages[1].tool_call_id",
            ),
        ];
        for (r, want) in cases {
            let (got, unsupported) = param(&all, &r);
            assert_eq!(got, want);
            assert!(!unsupported, "{want} should be invalid, not unsupported");
        }
    }

    #[test]
    fn test_validate_accepts_a_tool_round_trip() {
        let r = ChatRequest {
            tools: vec![weather()],
            messages: vec![
                Message::user("weather in Oslo?"),
                Message::Assistant(AssistantMessage {
                    tool_calls: vec![ToolCall {
                        id: "call_1".into(),
                        name: "get_weather".into(),
                        kind: ToolCallKind::Function,
                        input: "{\"city\":\"Oslo\"}".into(),
                    }],
                    ..AssistantMessage::default()
                }),
                Message::Tool {
                    tool_call_id: "call_1".into(),
                    content: "{\"temperature_c\":3}".into(),
                },
            ],
            ..request()
        };
        validate("m", &tools_caps(), &r).unwrap();
    }

    #[test]
    fn test_validate_refuses_what_the_model_cannot_do() {
        let image = Message::User {
            content: vec![
                ContentPart::Text("what is this?".into()),
                ContentPart::Image {
                    url: "https://example.com/cat.png".into(),
                    detail: ImageDetail::Auto,
                },
            ],
            name: None,
        };
        let cases: Vec<(ChatRequest, &str)> = vec![
            (ChatRequest { n: 2, ..request() }, "n"),
            (
                ChatRequest {
                    max_completion_tokens: Some(4096),
                    ..request()
                },
                "max_completion_tokens",
            ),
            (
                ChatRequest {
                    sampling: Sampling {
                        temperature: Some(0.7),
                        ..Sampling::default()
                    },
                    ..request()
                },
                "temperature",
            ),
            (
                ChatRequest {
                    sampling: Sampling {
                        stop: vec!["\n".into()],
                        ..Sampling::default()
                    },
                    ..request()
                },
                "stop",
            ),
            (
                ChatRequest {
                    logprobs: Some(Logprobs::default()),
                    ..request()
                },
                "logprobs",
            ),
            (
                ChatRequest {
                    tools: vec![weather()],
                    ..request()
                },
                "tools[0]",
            ),
            (
                ChatRequest {
                    response_format: ResponseFormat::JsonObject,
                    ..request()
                },
                "response_format",
            ),
            (
                ChatRequest {
                    messages: vec![image],
                    ..request()
                },
                "messages[0].content[1]",
            ),
            (
                ChatRequest {
                    modalities: vec![Modality::Text, Modality::Audio],
                    audio: Some(crate::chat::request::AudioOutput {
                        voice: "alloy".into(),
                        format: AudioFormat::Wav,
                    }),
                    ..request()
                },
                "modalities",
            ),
            (
                ChatRequest {
                    reasoning_effort: Some(crate::chat::request::ReasoningEffort::Low),
                    ..request()
                },
                "reasoning_effort",
            ),
        ];
        for (r, want) in cases {
            let (got, unsupported) = param(&Capabilities::default(), &r);
            assert_eq!(got, want);
            assert!(unsupported, "{want} should be unsupported, not invalid");
        }
    }

    #[test]
    fn test_validate_lets_a_greedy_model_take_the_greedy_settings() {
        let r = ChatRequest {
            sampling: Sampling {
                temperature: Some(0.0),
                top_p: Some(1.0),
                presence_penalty: Some(0.0),
                seed: Some(42),
                ..Sampling::default()
            },
            ..request()
        };
        validate("m", &Capabilities::default(), &r).unwrap();
    }

    #[test]
    fn test_validate_refuses_forced_and_parallel_calls_a_model_cannot_make() {
        let base = ChatRequest {
            tools: vec![weather()],
            ..request()
        };
        let forced = ChatRequest {
            tool_choice: ToolChoice::Required,
            ..base.clone()
        };
        assert_eq!(param(&tools_caps(), &forced), ("tool_choice".into(), true));
        let parallel = ChatRequest {
            parallel_tool_calls: Some(true),
            ..base.clone()
        };
        assert_eq!(
            param(&tools_caps(), &parallel),
            ("parallel_tool_calls".into(), true)
        );
        let able = Capabilities {
            tools: ToolSupport {
                functions: true,
                parallel: true,
                forced: true,
                ..ToolSupport::default()
            },
            ..Capabilities::default()
        };
        validate("m", &able, &forced).unwrap();
        validate("m", &able, &parallel).unwrap();
    }

    #[test]
    fn test_validate_names_the_model_in_an_unsupported_error() {
        let err = validate(
            "teeny-echo",
            &Capabilities::default(),
            &ChatRequest { n: 3, ..request() },
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("teeny-echo") && err.contains("`n`"), "{err}");
    }
}
