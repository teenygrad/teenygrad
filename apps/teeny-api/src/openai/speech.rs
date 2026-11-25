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

//! `/v1/audio/speech`: `CreateSpeechRequest` and the speech stream events.

use serde::{Deserialize, Serialize};
use teeny_audio::codec::AudioEncoding;
use teeny_audio::speech::{SpeechRequest, SpeechUsage};

use super::base64;
use super::error::ApiError;

/// A voice: by name, or a custom voice by id.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum Voice {
    Name(String),
    Custom { id: String },
}

/// `CreateSpeechRequest`.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CreateSpeechRequest {
    pub model: String,
    pub input: String,
    pub voice: Voice,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub response_format: Option<String>,
    #[serde(default)]
    pub speed: Option<f32>,
    #[serde(default)]
    pub stream_format: Option<String>,
}

/// How the audio is sent.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StreamFormat {
    /// The audio file as the response body.
    #[default]
    Audio,
    /// Server-sent events carrying base64 chunks.
    Sse,
}

impl CreateSpeechRequest {
    /// The core request, and how to send its audio.
    pub fn into_core(self) -> Result<(SpeechRequest, StreamFormat), ApiError> {
        let format = match self.response_format.as_deref().unwrap_or("mp3") {
            "mp3" => AudioEncoding::Mp3,
            "opus" => AudioEncoding::Opus,
            "aac" => AudioEncoding::Aac,
            "flac" => AudioEncoding::Flac,
            "wav" => AudioEncoding::Wav,
            "pcm" => AudioEncoding::Pcm,
            other => {
                return Err(ApiError::invalid_request(
                    format!(
                        "`{other}` is not an audio format; use mp3, opus, aac, flac, wav or pcm"
                    ),
                    Some("response_format"),
                ));
            }
        };
        let stream = match self.stream_format.as_deref().unwrap_or("audio") {
            "audio" => StreamFormat::Audio,
            "sse" => StreamFormat::Sse,
            other => {
                return Err(ApiError::invalid_request(
                    format!("`{other}` is not a stream format; use audio or sse"),
                    Some("stream_format"),
                ));
            }
        };
        let voice = match self.voice {
            Voice::Name(v) | Voice::Custom { id: v } => v,
        };
        Ok((
            SpeechRequest {
                model: self.model,
                input: self.input,
                voice,
                instructions: self.instructions,
                format,
                speed: self.speed.unwrap_or(1.0),
                stream: stream == StreamFormat::Sse,
            },
            stream,
        ))
    }
}

/// `SpeechAudioDeltaEvent`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AudioDeltaEvent {
    #[serde(rename = "type")]
    pub kind: &'static str,
    /// Base64 of the chunk's bytes.
    pub audio: String,
}

impl AudioDeltaEvent {
    pub fn new(bytes: &[u8]) -> Self {
        Self {
            kind: "speech.audio.delta",
            audio: base64::encode(bytes),
        }
    }
}

/// The usage `SpeechAudioDoneEvent` carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SpeechUsageJson {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

/// `SpeechAudioDoneEvent`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AudioDoneEvent {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub usage: SpeechUsageJson,
}

impl AudioDoneEvent {
    pub fn new(u: SpeechUsage) -> Self {
        Self {
            kind: "speech.audio.done",
            usage: SpeechUsageJson {
                input_tokens: u.input_tokens,
                output_tokens: u.output_tokens,
                total_tokens: u.input_tokens.saturating_add(u.output_tokens),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(v: serde_json::Value) -> Result<(SpeechRequest, StreamFormat), ApiError> {
        serde_json::from_value::<CreateSpeechRequest>(v)
            .unwrap()
            .into_core()
    }

    #[test]
    fn test_into_core_maps_formats_voices_and_defaults() {
        let (r, s) =
            parse(serde_json::json!({"model": "m", "input": "hi", "voice": "alloy"})).unwrap();
        assert_eq!(
            (r.format, r.speed, r.stream, s),
            (AudioEncoding::Mp3, 1.0, false, StreamFormat::Audio)
        );
        let (r, s) = parse(serde_json::json!({
            "model": "m", "input": "hi", "voice": {"id": "voice_123"},
            "response_format": "wav", "speed": 1.5, "stream_format": "sse"
        }))
        .unwrap();
        assert_eq!(
            (r.voice.as_str(), r.format, r.stream, s),
            ("voice_123", AudioEncoding::Wav, true, StreamFormat::Sse)
        );
        let bad = parse(
            serde_json::json!({"model": "m", "input": "hi", "voice": "a", "response_format": "ogg"}),
        );
        assert_eq!(
            bad.unwrap_err().body.param.as_deref(),
            Some("response_format")
        );
    }
}
