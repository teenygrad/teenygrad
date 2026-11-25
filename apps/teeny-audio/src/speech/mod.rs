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

//! Text to speech (`/v1/audio/speech`).
//!
//! A model implements [`SpeechModel`]: it declares its [`SpeechCapabilities`] and speaks
//! the input as mono samples at its own rate ([`SpeechStream`]). Encoding them into the
//! requested format is shared ([`encode_stream`], over [`crate::codec`]), so no model
//! repeats it; [`validate`] refuses what the model cannot do first.
//! [`EchoSpeechModel`] is the reference implementation.

mod echo;

pub use echo::EchoSpeechModel;

use std::pin::Pin;

use futures::future::BoxFuture;
use futures::{Stream, StreamExt};

use crate::codec::{self, AudioEncoding};
use crate::{Error, ModelInfo, Result};

/// The fastest and slowest speech any model is asked for.
pub const SPEED_RANGE: (f32, f32) = (0.25, 4.0);

/// A request to speak some text.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechRequest {
    pub model: String,
    pub input: String,
    pub voice: String,
    /// How to speak it (tone, accent, pace), for models that take directions.
    pub instructions: Option<String>,
    pub format: AudioEncoding,
    /// 1 is normal.
    pub speed: f32,
    /// Whether the audio is wanted as it is produced.
    pub stream: bool,
}

impl Default for SpeechRequest {
    fn default() -> Self {
        Self {
            model: String::new(),
            input: String::new(),
            voice: String::new(),
            instructions: None,
            format: AudioEncoding::Mp3,
            speed: 1.0,
            stream: false,
        }
    }
}

/// What speaking used.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SpeechUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// A model's output: mono samples at its rate, then its usage.
#[derive(Debug, Clone, PartialEq)]
pub enum SpeechEvent {
    Samples(Vec<f32>),
    Done(SpeechUsage),
}

/// A speech model's output.
pub type SpeechStream = Pin<Box<dyn Stream<Item = Result<SpeechEvent>> + Send>>;

/// Encoded audio, then the usage.
#[derive(Debug, Clone, PartialEq)]
pub enum SpeechChunk {
    Audio(Vec<u8>),
    Done(SpeechUsage),
}

/// Speech in the requested format.
pub type AudioStream = Pin<Box<dyn Stream<Item = Result<SpeechChunk>> + Send>>;

/// What a speech model can do.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeechCapabilities {
    /// The voices it speaks in; the first is its default.
    pub voices: Vec<String>,
    /// The slowest and fastest it speaks, within [`SPEED_RANGE`].
    pub speed: (f32, f32),
    pub instructions: bool,
    pub streaming: bool,
    /// The longest input, in characters.
    pub max_input_chars: usize,
    /// The rate of the mono samples it produces.
    pub sample_rate: u32,
}

impl Default for SpeechCapabilities {
    fn default() -> Self {
        Self {
            voices: Vec::new(),
            speed: (1.0, 1.0),
            instructions: false,
            streaming: false,
            max_input_chars: 4096,
            sample_rate: 24_000,
        }
    }
}

fn invalid(parameter: &str, detail: impl Into<String>) -> Error {
    Error::InvalidParameter {
        parameter: parameter.to_owned(),
        detail: detail.into(),
    }
}

fn unsupported(model: &str, parameter: &str, detail: impl Into<String>) -> Error {
    Error::UnsupportedParameter {
        model: model.to_owned(),
        parameter: parameter.to_owned(),
        detail: detail.into(),
    }
}

/// Checks `request` for the model `model` with capabilities `caps`.
pub fn validate(model: &str, caps: &SpeechCapabilities, r: &SpeechRequest) -> Result<()> {
    if r.input.trim().is_empty() {
        return Err(invalid("input", "there is no text to speak"));
    }
    let chars = r.input.chars().count();
    if chars > caps.max_input_chars {
        return Err(invalid(
            "input",
            format!(
                "{chars} characters; the model speaks at most {}",
                caps.max_input_chars
            ),
        ));
    }
    if !caps.voices.iter().any(|v| v == &r.voice) {
        return Err(unsupported(
            model,
            "voice",
            format!(
                "the voice `{}`; it speaks in {}",
                r.voice,
                caps.voices.join(", ")
            ),
        ));
    }
    let (lo, hi) = SPEED_RANGE;
    if !(lo..=hi).contains(&r.speed) {
        return Err(invalid(
            "speed",
            format!("{} is outside {lo} to {hi}", r.speed),
        ));
    }
    let (lo, hi) = caps.speed;
    if !(lo..=hi).contains(&r.speed) {
        return Err(unsupported(
            model,
            "speed",
            format!("speed {}; it speaks at {lo} to {hi}", r.speed),
        ));
    }
    if r.instructions.is_some() && !caps.instructions {
        return Err(unsupported(model, "instructions", "speaking instructions"));
    }
    if r.stream && !caps.streaming {
        return Err(unsupported(model, "stream_format", "streamed speech"));
    }
    if !r.format.encodable() {
        return Err(unsupported(
            model,
            "response_format",
            format!(
                "{} output (only wav and pcm are encoded so far)",
                r.format.name()
            ),
        ));
    }
    Ok(())
}

/// A model that turns text into speech.
pub trait SpeechModel: Send + Sync + 'static {
    fn info(&self) -> ModelInfo;

    fn capabilities(&self) -> SpeechCapabilities;

    /// Speaks `request`, which has passed [`validate`], as mono samples at
    /// [`SpeechCapabilities::sample_rate`]. The stream ends with [`SpeechEvent::Done`].
    fn speak(&self, request: SpeechRequest) -> BoxFuture<'_, Result<SpeechStream>>;
}

/// Encodes a model's samples (mono, at `sample_rate`) as `format`.
///
/// Streamed, each batch of samples is encoded as it comes (a WAV stream opens with a
/// header whose lengths are left open); otherwise the samples are gathered and encoded
/// once, so a WAV file's header has its exact length.
pub fn encode_stream(
    stream: SpeechStream,
    format: AudioEncoding,
    sample_rate: u32,
    streamed: bool,
) -> AudioStream {
    if streamed {
        let mut header =
            (format == AudioEncoding::Wav).then(|| codec::wav_header(sample_rate, 1, None));
        return Box::pin(stream.map(move |event| {
            Ok(match event? {
                SpeechEvent::Samples(s) => {
                    let mut bytes = header.take().unwrap_or_default();
                    bytes.extend_from_slice(&codec::pcm16(&s));
                    SpeechChunk::Audio(bytes)
                }
                SpeechEvent::Done(u) => SpeechChunk::Done(u),
            })
        }));
    }
    Box::pin(
        futures::stream::once(async move {
            let mut samples = Vec::new();
            let mut usage = SpeechUsage::default();
            let mut stream = stream;
            while let Some(event) = stream.next().await {
                match event? {
                    SpeechEvent::Samples(s) => samples.extend_from_slice(&s),
                    SpeechEvent::Done(u) => usage = u,
                }
            }
            let pcm = codec::Pcm {
                sample_rate,
                channels: 1,
                samples,
            };
            Ok::<_, Error>((codec::encode(&pcm, format)?, usage))
        })
        .flat_map(|r| {
            futures::stream::iter(match r {
                Ok((bytes, usage)) => {
                    vec![Ok(SpeechChunk::Audio(bytes)), Ok(SpeechChunk::Done(usage))]
                }
                Err(e) => vec![Err(e)],
            })
        }),
    )
}

/// The audio and usage of a stream, gathered.
pub async fn collect(mut stream: AudioStream) -> Result<(Vec<u8>, SpeechUsage)> {
    let mut bytes = Vec::new();
    let mut usage = SpeechUsage::default();
    while let Some(chunk) = stream.next().await {
        match chunk? {
            SpeechChunk::Audio(b) => bytes.extend_from_slice(&b),
            SpeechChunk::Done(u) => usage = u,
        }
    }
    Ok((bytes, usage))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps() -> SpeechCapabilities {
        SpeechCapabilities {
            voices: vec!["alloy".into()],
            speed: (0.5, 2.0),
            ..SpeechCapabilities::default()
        }
    }

    fn request() -> SpeechRequest {
        SpeechRequest {
            model: "m".into(),
            input: "hello".into(),
            voice: "alloy".into(),
            format: AudioEncoding::Wav,
            ..SpeechRequest::default()
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
        assert!(validate("m", &caps(), &request()).is_ok());
        let cases = [
            (
                SpeechRequest {
                    input: " ".into(),
                    ..request()
                },
                "input",
            ),
            (
                SpeechRequest {
                    input: "x".repeat(5000),
                    ..request()
                },
                "input",
            ),
            (
                SpeechRequest {
                    voice: "nova".into(),
                    ..request()
                },
                "voice",
            ),
            (
                SpeechRequest {
                    speed: 9.0,
                    ..request()
                },
                "speed",
            ),
            (
                SpeechRequest {
                    speed: 3.0,
                    ..request()
                },
                "speed",
            ),
            (
                SpeechRequest {
                    instructions: Some("whisper".into()),
                    ..request()
                },
                "instructions",
            ),
            (
                SpeechRequest {
                    stream: true,
                    ..request()
                },
                "stream_format",
            ),
            (
                SpeechRequest {
                    format: AudioEncoding::Mp3,
                    ..request()
                },
                "response_format",
            ),
        ];
        for (r, want) in cases {
            assert_eq!(param(validate("m", &caps(), &r)), want);
        }
    }

    fn samples() -> SpeechStream {
        Box::pin(futures::stream::iter([
            Ok(SpeechEvent::Samples(vec![0.5; 3])),
            Ok(SpeechEvent::Samples(vec![-0.5; 2])),
            Ok(SpeechEvent::Done(SpeechUsage {
                input_tokens: 1,
                output_tokens: 2,
            })),
        ]))
    }

    #[tokio::test]
    async fn test_encode_stream_writes_whole_and_streamed_wav() {
        let (whole, usage) = collect(encode_stream(samples(), AudioEncoding::Wav, 8000, false))
            .await
            .unwrap();
        assert_eq!(usage.output_tokens, 2);
        let pcm = codec::decode_wav(&whole).unwrap();
        assert_eq!((pcm.sample_rate, pcm.samples.len()), (8000, 5));

        let chunks: Vec<SpeechChunk> = encode_stream(samples(), AudioEncoding::Wav, 8000, true)
            .map(|c| c.unwrap())
            .collect()
            .await;
        let SpeechChunk::Audio(first) = &chunks[0] else {
            panic!("audio first")
        };
        assert_eq!((&first[..4], first.len()), (&b"RIFF"[..], 44 + 6));
        assert_eq!(chunks[1], SpeechChunk::Audio(codec::pcm16(&[-0.5; 2])));
        let (raw, _) = collect(encode_stream(samples(), AudioEncoding::Pcm, 8000, true))
            .await
            .unwrap();
        assert_eq!(raw.len(), 10);
    }
}
