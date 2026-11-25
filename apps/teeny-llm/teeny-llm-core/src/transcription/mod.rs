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

//! Speech to text (`/v1/audio/transcriptions` and `/translations`), for Whisper-class
//! models.
//!
//! A model implements [`TranscriptionModel`]: it declares its
//! [`TranscriptionCapabilities`] and turns decoded audio into a [`TranscriptionStream`].
//! The recording is decoded, mixed to mono and resampled to the model's rate by the
//! engine (see [`crate::audio`]), so no model repeats that; [`validate`] refuses what the
//! model cannot do first. Translation is the same call with English output.
//! [`EchoTranscriptionModel`] is the reference implementation.

mod echo;

pub use echo::EchoTranscriptionModel;

use std::pin::Pin;

use futures::future::BoxFuture;
use futures::{Stream, StreamExt};

use crate::audio::{AudioContainer, Pcm};
use crate::{Error, ModelInfo, Result};

/// Transcribe in the spoken language, or translate into English.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TranscriptionTask {
    #[default]
    Transcribe,
    Translate,
}

/// Which timestamps a verbose reply carries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Timestamps {
    pub segments: bool,
    pub words: bool,
}

/// A request to transcribe (or translate) one recording.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TranscriptionRequest {
    pub model: String,
    pub task: TranscriptionTask,
    /// The recording, as uploaded.
    pub audio: Vec<u8>,
    pub container: Option<AudioContainer>,
    /// The spoken language (ISO 639-1), if known.
    pub language: Option<String>,
    /// Text that came before, to guide spelling and style.
    pub prompt: Option<String>,
    /// 0 to 1.
    pub temperature: Option<f32>,
    pub timestamps: Timestamps,
    /// Whether to return the log probability of each output token.
    pub logprobs: bool,
    /// Whether the text is wanted as it is produced.
    pub stream: bool,
}

/// A stretch of the transcript, with its times in seconds.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Segment {
    pub id: u32,
    pub start: f32,
    pub end: f32,
    pub text: String,
}

/// A word, with its times in seconds.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Word {
    pub word: String,
    pub start: f32,
    pub end: f32,
}

/// What a transcription used.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TranscriptionUsage {
    /// The recording's length.
    pub seconds: f32,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// A whole transcript.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Transcription {
    pub text: String,
    /// The language detected (or given).
    pub language: Option<String>,
    pub duration: f32,
    pub segments: Vec<Segment>,
    pub words: Vec<Word>,
    /// Per output token, when asked for.
    pub logprobs: Vec<(String, f32)>,
    pub usage: TranscriptionUsage,
}

/// Transcript text as it is produced, then the whole transcript.
#[derive(Debug, Clone, PartialEq)]
pub enum TranscriptionEvent {
    Delta(String),
    Done(Transcription),
}

/// A transcription model's output.
pub type TranscriptionStream = Pin<Box<dyn Stream<Item = Result<TranscriptionEvent>> + Send>>;

/// What a transcription model can do.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptionCapabilities {
    /// The languages it hears (ISO 639-1); empty means any.
    pub languages: Vec<String>,
    pub translation: bool,
    pub timestamps: Timestamps,
    pub streaming: bool,
    pub logprobs: bool,
    /// Longest recording, in seconds.
    pub max_seconds: f32,
    /// The rate it takes mono samples at.
    pub sample_rate: u32,
}

impl Default for TranscriptionCapabilities {
    fn default() -> Self {
        Self {
            languages: Vec::new(),
            translation: false,
            timestamps: Timestamps::default(),
            streaming: false,
            logprobs: false,
            max_seconds: 30.0,
            sample_rate: 16_000,
        }
    }
}

fn unsupported(model: &str, parameter: &str, detail: impl Into<String>) -> Error {
    Error::UnsupportedParameter {
        model: model.to_owned(),
        parameter: parameter.to_owned(),
        detail: detail.into(),
    }
}

/// Checks `request` for the model `model` with capabilities `caps`, before the audio is
/// decoded. Its length is checked against `max_seconds` once it is ([`check_duration`]).
pub fn validate(
    model: &str,
    caps: &TranscriptionCapabilities,
    r: &TranscriptionRequest,
) -> Result<()> {
    if r.audio.is_empty() {
        return Err(Error::InvalidParameter {
            parameter: "file".into(),
            detail: "the recording is empty".into(),
        });
    }
    if r.container.is_none() {
        return Err(Error::InvalidParameter {
            parameter: "file".into(),
            detail: "the file name's extension does not name an audio format (flac, mp3, mp4, mpeg, mpga, \
                     m4a, ogg, wav, webm)"
                .into(),
        });
    }
    if let Some(t) = r.temperature
        && !(0.0..=1.0).contains(&t)
    {
        return Err(Error::InvalidParameter {
            parameter: "temperature".into(),
            detail: format!("{t} is outside 0 to 1"),
        });
    }
    if r.task == TranscriptionTask::Translate && !caps.translation {
        return Err(unsupported(model, "model", "translation into English"));
    }
    if let Some(l) = &r.language
        && !caps.languages.is_empty()
        && !caps.languages.iter().any(|c| c.eq_ignore_ascii_case(l))
    {
        return Err(unsupported(
            model,
            "language",
            format!("`{l}`; it hears {}", caps.languages.join(", ")),
        ));
    }
    if r.timestamps.segments && !caps.timestamps.segments {
        return Err(unsupported(
            model,
            "timestamp_granularities",
            "segment timestamps",
        ));
    }
    if r.timestamps.words && !caps.timestamps.words {
        return Err(unsupported(
            model,
            "timestamp_granularities",
            "word timestamps",
        ));
    }
    if r.logprobs && !caps.logprobs {
        return Err(unsupported(model, "include", "log probabilities"));
    }
    if r.stream && !caps.streaming {
        return Err(unsupported(model, "stream", "streamed transcripts"));
    }
    Ok(())
}

/// Refuses a recording longer than the model takes.
pub fn check_duration(caps: &TranscriptionCapabilities, audio: &Pcm) -> Result<()> {
    let d = audio.duration();
    if d > caps.max_seconds {
        return Err(Error::InvalidParameter {
            parameter: "file".into(),
            detail: format!(
                "{d:.1} s of audio; the model takes at most {:.1} s",
                caps.max_seconds
            ),
        });
    }
    Ok(())
}

/// A model that turns speech into text.
pub trait TranscriptionModel: Send + Sync + 'static {
    fn info(&self) -> ModelInfo;

    fn capabilities(&self) -> TranscriptionCapabilities;

    /// Transcribes `audio` (mono, at [`TranscriptionCapabilities::sample_rate`]) for
    /// `request`, which has passed [`validate`]. The stream ends with
    /// [`TranscriptionEvent::Done`].
    fn transcribe(
        &self,
        request: TranscriptionRequest,
        audio: Pcm,
    ) -> BoxFuture<'_, Result<TranscriptionStream>>;
}

/// The transcript a stream ends with.
pub async fn collect(mut stream: TranscriptionStream) -> Result<Transcription> {
    while let Some(event) = stream.next().await {
        if let TranscriptionEvent::Done(t) = event? {
            return Ok(t);
        }
    }
    Err(Error::Inference(
        "the transcription stream ended without its transcript".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> TranscriptionRequest {
        TranscriptionRequest {
            model: "m".into(),
            audio: vec![0],
            container: Some(AudioContainer::Wav),
            ..TranscriptionRequest::default()
        }
    }

    fn param(r: Result<()>) -> String {
        match r {
            Err(Error::InvalidParameter { parameter, .. })
            | Err(Error::UnsupportedParameter { parameter, .. }) => parameter,
            other => panic!("not refused by parameter: {other:?}"),
        }
    }

    #[test]
    fn test_validate_refuses_what_the_model_cannot_do() {
        let caps = TranscriptionCapabilities {
            languages: vec!["en".into()],
            ..TranscriptionCapabilities::default()
        };
        assert!(validate("m", &caps, &request()).is_ok());
        let cases = [
            (
                TranscriptionRequest {
                    audio: vec![],
                    ..request()
                },
                "file",
            ),
            (
                TranscriptionRequest {
                    container: None,
                    ..request()
                },
                "file",
            ),
            (
                TranscriptionRequest {
                    temperature: Some(1.5),
                    ..request()
                },
                "temperature",
            ),
            (
                TranscriptionRequest {
                    task: TranscriptionTask::Translate,
                    ..request()
                },
                "model",
            ),
            (
                TranscriptionRequest {
                    language: Some("fr".into()),
                    ..request()
                },
                "language",
            ),
            (
                TranscriptionRequest {
                    timestamps: Timestamps {
                        words: true,
                        segments: false,
                    },
                    ..request()
                },
                "timestamp_granularities",
            ),
            (
                TranscriptionRequest {
                    logprobs: true,
                    ..request()
                },
                "include",
            ),
            (
                TranscriptionRequest {
                    stream: true,
                    ..request()
                },
                "stream",
            ),
        ];
        for (r, want) in cases {
            assert_eq!(param(validate("m", &caps, &r)), want);
        }
        let long = Pcm {
            sample_rate: 10,
            channels: 1,
            samples: vec![0.0; 400],
        };
        assert!(check_duration(&caps, &long).is_err());
    }
}
