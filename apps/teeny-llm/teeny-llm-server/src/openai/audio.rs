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

//! `/v1/audio/transcriptions` and `/v1/audio/translations`: the multipart form, and the
//! transcript in each `response_format`.

use serde::Serialize;
use teeny_llm_core::audio::AudioContainer;
use teeny_llm_core::transcription::{
    Segment, Timestamps, Transcription, TranscriptionRequest, TranscriptionTask, Word,
};

use super::error::ApiError;

/// One part of a multipart form: its name, the file name it was sent with, and its bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormPart {
    pub name: String,
    pub file_name: Option<String>,
    pub data: Vec<u8>,
}

/// How the transcript is returned.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ResponseFormat {
    #[default]
    Json,
    Text,
    Srt,
    VerboseJson,
    Vtt,
}

impl ResponseFormat {
    fn parse(s: &str) -> Result<Self, ApiError> {
        Ok(match s {
            "json" => Self::Json,
            "text" => Self::Text,
            "srt" => Self::Srt,
            "verbose_json" => Self::VerboseJson,
            "vtt" => Self::Vtt,
            other => {
                return Err(ApiError::invalid_request(
                    format!(
                        "`{other}` is not a response format; use json, text, srt, verbose_json or vtt"
                    ),
                    Some("response_format"),
                ));
            }
        })
    }
}

fn text(part: &FormPart) -> Result<String, ApiError> {
    String::from_utf8(part.data.clone()).map_err(|_| {
        ApiError::invalid_request(
            format!("the `{}` field is not UTF-8 text", part.name),
            Some(&part.name),
        )
    })
}

fn boolean(part: &FormPart) -> Result<bool, ApiError> {
    match text(part)?.trim() {
        "true" | "1" => Ok(true),
        "false" | "0" => Ok(false),
        other => Err(ApiError::invalid_request(
            format!("`{other}` is not a boolean"),
            Some(&part.name),
        )),
    }
}

/// The request a transcription (`task` [`TranscriptionTask::Transcribe`]) or translation
/// form describes, and the format to answer in.
pub fn parse_form(
    task: TranscriptionTask,
    parts: Vec<FormPart>,
) -> Result<(TranscriptionRequest, ResponseFormat), ApiError> {
    let mut request = TranscriptionRequest {
        task,
        ..TranscriptionRequest::default()
    };
    let mut format = ResponseFormat::Json;
    let mut granularities: Vec<String> = Vec::new();
    let mut have_file = false;
    let mut have_model = false;
    for part in parts {
        // Array fields arrive as `name[]` from most clients.
        let name = part.name.trim_end_matches("[]");
        let translate = task == TranscriptionTask::Translate;
        match name {
            "file" => {
                request.container = part
                    .file_name
                    .as_deref()
                    .and_then(AudioContainer::from_extension);
                request.audio = part.data;
                have_file = true;
            }
            "model" => {
                request.model = text(&part)?;
                have_model = true;
            }
            "language" if !translate => request.language = Some(text(&part)?),
            "prompt" => request.prompt = Some(text(&part)?),
            "response_format" => format = ResponseFormat::parse(text(&part)?.trim())?,
            "temperature" => {
                let t = text(&part)?;
                request.temperature = Some(t.trim().parse().map_err(|_| {
                    ApiError::invalid_request(format!("`{t}` is not a number"), Some("temperature"))
                })?);
            }
            "timestamp_granularities" if !translate => granularities.push(text(&part)?),
            "include" if !translate => match text(&part)?.as_str() {
                "logprobs" => request.logprobs = true,
                other => {
                    return Err(ApiError::invalid_request(
                        format!("`{other}` cannot be included; only `logprobs` can"),
                        Some("include"),
                    ));
                }
            },
            "stream" if !translate => request.stream = boolean(&part)?,
            "chunking_strategy" if !translate => {
                let s = text(&part)?;
                if s.trim() != "auto" {
                    return Err(ApiError::invalid_request(
                        "only the `auto` chunking strategy is supported",
                        Some("chunking_strategy"),
                    ));
                }
            }
            // Fields this server does not act on are ignored, as for JSON bodies.
            _ => {}
        }
    }
    if !have_file {
        return Err(ApiError::invalid_request(
            "the form has no `file` field",
            Some("file"),
        ));
    }
    if !have_model {
        return Err(ApiError::invalid_request(
            "the form has no `model` field",
            Some("model"),
        ));
    }
    for g in &granularities {
        match g.as_str() {
            "segment" => request.timestamps.segments = true,
            "word" => request.timestamps.words = true,
            other => {
                return Err(ApiError::invalid_request(
                    format!("`{other}` is not a timestamp granularity; use word or segment"),
                    Some("timestamp_granularities"),
                ));
            }
        }
    }
    if !granularities.is_empty() && format != ResponseFormat::VerboseJson {
        return Err(ApiError::invalid_request(
            "timestamp_granularities needs response_format verbose_json",
            Some("timestamp_granularities"),
        ));
    }
    // Verbose output carries segments unless only words were asked for; subtitles need them.
    match format {
        ResponseFormat::VerboseJson if granularities.is_empty() => {
            request.timestamps.segments = true
        }
        ResponseFormat::Srt | ResponseFormat::Vtt => request.timestamps.segments = true,
        _ => {}
    }
    if request.stream && !matches!(format, ResponseFormat::Json | ResponseFormat::Text) {
        return Err(ApiError::invalid_request(
            "streamed transcripts are sent as json or text events only",
            Some("stream"),
        ));
    }
    Ok((request, format))
}

/// `TranscriptTextUsageDuration`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DurationUsage {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub seconds: f32,
}

/// One token's log probability.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TokenLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
}

/// `CreateTranscriptionResponseJson` (and the translation's).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TranscriptionJson {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Vec<TokenLogprob>>,
    pub usage: DurationUsage,
}

/// `TranscriptionSegment`. The decoder statistics this server does not track are zero.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SegmentJson {
    pub id: u32,
    pub seek: u32,
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub tokens: Vec<u32>,
    pub temperature: f32,
    pub avg_logprob: f32,
    pub compression_ratio: f32,
    pub no_speech_prob: f32,
}

/// `TranscriptionWord`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct WordJson {
    pub word: String,
    pub start: f32,
    pub end: f32,
}

/// `CreateTranscriptionResponseVerboseJson` (and the translation's).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TranscriptionVerboseJson {
    pub task: &'static str,
    pub language: String,
    pub duration: f32,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub words: Option<Vec<WordJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segments: Option<Vec<SegmentJson>>,
    pub usage: DurationUsage,
}

/// `TranscriptTextDeltaEvent`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TextDeltaEvent {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub delta: String,
}

/// `TranscriptTextDoneEvent`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TextDoneEvent {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Vec<TokenLogprob>>,
    pub usage: DurationUsage,
}

impl TextDeltaEvent {
    pub fn new(delta: String) -> Self {
        Self {
            kind: "transcript.text.delta",
            delta,
        }
    }
}

fn usage(t: &Transcription) -> DurationUsage {
    DurationUsage {
        kind: "duration",
        seconds: t.usage.seconds,
    }
}

fn logprobs(t: &Transcription, wanted: bool) -> Option<Vec<TokenLogprob>> {
    wanted.then(|| {
        t.logprobs
            .iter()
            .map(|(token, logprob)| TokenLogprob {
                bytes: token.as_bytes().to_vec(),
                token: token.clone(),
                logprob: *logprob,
            })
            .collect()
    })
}

impl TextDoneEvent {
    pub fn new(t: &Transcription, wanted_logprobs: bool) -> Self {
        Self {
            kind: "transcript.text.done",
            text: t.text.clone(),
            logprobs: logprobs(t, wanted_logprobs),
            usage: usage(t),
        }
    }
}

/// A finished transcript, in the format asked for: a JSON value, or a text body with its
/// media type.
#[derive(Debug, Clone, PartialEq)]
pub enum Rendered {
    Json(serde_json::Value),
    Text { mime: &'static str, body: String },
}

fn stamp(seconds: f32, decimal: char) -> String {
    let ms = (f64::from(seconds.max(0.0)) * 1000.0).round() as u64;
    format!(
        "{:02}:{:02}:{:02}{decimal}{:03}",
        ms / 3_600_000,
        ms / 60_000 % 60,
        ms / 1000 % 60,
        ms % 1000
    )
}

fn subtitles(segments: &[Segment], vtt: bool) -> String {
    let mut out = String::new();
    if vtt {
        out.push_str("WEBVTT\n\n");
    }
    let decimal = if vtt { '.' } else { ',' };
    for (i, s) in segments.iter().enumerate() {
        if !vtt {
            out.push_str(&format!("{}\n", i + 1));
        }
        out.push_str(&format!(
            "{} --> {}\n{}\n\n",
            stamp(s.start, decimal),
            stamp(s.end, decimal),
            s.text.trim()
        ));
    }
    out
}

fn json<T: Serialize>(value: &T) -> Rendered {
    // These plain structs always serialize.
    Rendered::Json(serde_json::to_value(value).unwrap_or(serde_json::Value::Null))
}

/// Renders `t` as `format` for a request `request`.
pub fn render(
    t: &Transcription,
    request: &TranscriptionRequest,
    format: ResponseFormat,
) -> Rendered {
    match format {
        ResponseFormat::Json => json(&TranscriptionJson {
            text: t.text.clone(),
            logprobs: logprobs(t, request.logprobs),
            usage: usage(t),
        }),
        ResponseFormat::VerboseJson => {
            let Timestamps { segments, words } = request.timestamps;
            json(&TranscriptionVerboseJson {
                task: match request.task {
                    TranscriptionTask::Transcribe => "transcribe",
                    TranscriptionTask::Translate => "translate",
                },
                language: t.language.clone().unwrap_or_default(),
                duration: t.duration,
                text: t.text.clone(),
                words: words.then(|| {
                    t.words
                        .iter()
                        .map(|Word { word, start, end }| WordJson {
                            word: word.clone(),
                            start: *start,
                            end: *end,
                        })
                        .collect()
                }),
                segments: segments.then(|| {
                    t.segments
                        .iter()
                        .map(|s| SegmentJson {
                            id: s.id,
                            seek: 0,
                            start: s.start,
                            end: s.end,
                            text: s.text.clone(),
                            tokens: Vec::new(),
                            temperature: request.temperature.unwrap_or(0.0),
                            avg_logprob: 0.0,
                            compression_ratio: 0.0,
                            no_speech_prob: 0.0,
                        })
                        .collect()
                }),
                usage: usage(t),
            })
        }
        ResponseFormat::Text => Rendered::Text {
            mime: "text/plain; charset=utf-8",
            body: format!("{}\n", t.text),
        },
        ResponseFormat::Srt => Rendered::Text {
            mime: "application/x-subrip",
            body: subtitles(&t.segments, false),
        },
        ResponseFormat::Vtt => Rendered::Text {
            mime: "text/vtt; charset=utf-8",
            body: subtitles(&t.segments, true),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part(name: &str, value: &str) -> FormPart {
        FormPart {
            name: name.into(),
            file_name: None,
            data: value.as_bytes().to_vec(),
        }
    }

    fn file() -> FormPart {
        FormPart {
            name: "file".into(),
            file_name: Some("talk.wav".into()),
            data: vec![1, 2],
        }
    }

    fn param(r: Result<(TranscriptionRequest, ResponseFormat), ApiError>) -> Option<String> {
        r.err().and_then(|e| e.body.param)
    }

    #[test]
    fn test_parse_form_reads_the_fields() {
        let (r, f) = parse_form(
            TranscriptionTask::Transcribe,
            vec![
                file(),
                part("model", "m"),
                part("language", "en"),
                part("response_format", "verbose_json"),
                part("timestamp_granularities[]", "word"),
                part("include[]", "logprobs"),
                part("temperature", "0.2"),
                part("unknown", "ignored"),
            ],
        )
        .unwrap();
        assert_eq!(f, ResponseFormat::VerboseJson);
        assert_eq!(r.container, Some(AudioContainer::Wav));
        assert_eq!((r.model.as_str(), r.language.as_deref()), ("m", Some("en")));
        assert_eq!(
            r.timestamps,
            Timestamps {
                segments: false,
                words: true
            }
        );
        assert!(r.logprobs);
        assert_eq!(r.temperature, Some(0.2));
    }

    #[test]
    fn test_parse_form_refuses_bad_fields_by_name() {
        let t = TranscriptionTask::Transcribe;
        assert_eq!(
            param(parse_form(t, vec![part("model", "m")])).as_deref(),
            Some("file")
        );
        assert_eq!(param(parse_form(t, vec![file()])).as_deref(), Some("model"));
        let with = |p: FormPart| parse_form(t, vec![file(), part("model", "m"), p]);
        assert_eq!(
            param(with(part("response_format", "xml"))).as_deref(),
            Some("response_format")
        );
        assert_eq!(
            param(with(part("timestamp_granularities[]", "word"))).as_deref(),
            Some("timestamp_granularities")
        );
        assert_eq!(
            param(with(part("include[]", "everything"))).as_deref(),
            Some("include")
        );
        assert_eq!(
            param(with(part("temperature", "warm"))).as_deref(),
            Some("temperature")
        );
        let streamed_srt = parse_form(
            t,
            vec![
                file(),
                part("model", "m"),
                part("stream", "true"),
                part("response_format", "srt"),
            ],
        );
        assert_eq!(param(streamed_srt).as_deref(), Some("stream"));
    }

    #[test]
    fn test_render_writes_subtitles_and_json() {
        let t = Transcription {
            text: "hello there".into(),
            duration: 62.5,
            segments: vec![Segment {
                id: 0,
                start: 61.0,
                end: 62.5,
                text: " hello there".into(),
            }],
            ..Transcription::default()
        };
        let r = TranscriptionRequest::default();
        let Rendered::Text { body, .. } = render(&t, &r, ResponseFormat::Srt) else {
            panic!("srt is text");
        };
        assert_eq!(body, "1\n00:01:01,000 --> 00:01:02,500\nhello there\n\n");
        let Rendered::Text { body, .. } = render(&t, &r, ResponseFormat::Vtt) else {
            panic!("vtt is text");
        };
        assert!(body.starts_with("WEBVTT\n\n00:01:01.000 --> 00:01:02.500\n"));
        let Rendered::Json(v) = render(&t, &r, ResponseFormat::Json) else {
            panic!("json is json");
        };
        assert_eq!(v["text"], "hello there");
        assert_eq!(v["usage"]["type"], "duration");
        assert!(v.get("logprobs").is_none());
    }
}
