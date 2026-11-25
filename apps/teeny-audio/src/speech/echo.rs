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

//! [`EchoSpeechModel`], the reference [`SpeechModel`].

use futures::future::BoxFuture;

use super::{
    SpeechCapabilities, SpeechEvent, SpeechModel, SpeechRequest, SpeechStream, SpeechUsage,
};
use crate::{ModelInfo, Result};

/// A speech model without weights: it "speaks" each word as a short tone, its pitch set
/// by the voice, followed by a pause, one batch of samples per word. At speed 1 a word
/// takes 0.15 s and a pause 0.05 s.
#[derive(Debug, Clone, Copy, Default)]
pub struct EchoSpeechModel;

impl EchoSpeechModel {
    pub const MODEL_ID: &'static str = "teeny-echo-speak";

    /// Its voices, low to high.
    pub const VOICES: [&'static str; 3] = ["alloy", "echo", "nova"];

    /// Its sample rate, OpenAI's for raw PCM.
    pub const SAMPLE_RATE: u32 = 24_000;
}

impl SpeechModel for EchoSpeechModel {
    fn info(&self) -> ModelInfo {
        ModelInfo {
            id: Self::MODEL_ID.to_owned(),
            created: 0,
            owned_by: "teenygrad".to_owned(),
        }
    }

    fn capabilities(&self) -> SpeechCapabilities {
        SpeechCapabilities {
            voices: Self::VOICES.iter().map(|v| (*v).to_owned()).collect(),
            speed: super::SPEED_RANGE,
            instructions: true,
            streaming: true,
            max_input_chars: 4096,
            sample_rate: Self::SAMPLE_RATE,
        }
    }

    fn speak(&self, request: SpeechRequest) -> BoxFuture<'_, Result<SpeechStream>> {
        Box::pin(async move {
            let rate = Self::SAMPLE_RATE as f32;
            let voice = Self::VOICES
                .iter()
                .position(|v| *v == request.voice)
                .unwrap_or(0);
            let hz = 220.0 * (voice + 1) as f32;
            let tone = (0.15 * rate / request.speed) as usize;
            let pause = (0.05 * rate / request.speed) as usize;
            let words = request.input.split_whitespace().count();
            let mut events: Vec<Result<SpeechEvent>> = (0..words)
                .map(|_| {
                    let mut s: Vec<f32> = (0..tone)
                        .map(|i| (std::f32::consts::TAU * hz * i as f32 / rate).sin() * 0.3)
                        .collect();
                    s.resize(tone + pause, 0.0);
                    Ok(SpeechEvent::Samples(s))
                })
                .collect();
            let n = u32::try_from(words).unwrap_or(u32::MAX);
            events.push(Ok(SpeechEvent::Done(SpeechUsage {
                input_tokens: n,
                output_tokens: n,
            })));
            let stream: SpeechStream = Box::pin(futures::stream::iter(events));
            Ok(stream)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::AudioEncoding;
    use crate::speech::{collect, encode_stream, validate};

    #[tokio::test]
    async fn test_echo_speaks_a_tone_per_word_at_the_requested_speed() {
        let m = EchoSpeechModel;
        let request = SpeechRequest {
            input: "two words".into(),
            voice: "nova".into(),
            speed: 2.0,
            format: AudioEncoding::Pcm,
            ..SpeechRequest::default()
        };
        validate(EchoSpeechModel::MODEL_ID, &m.capabilities(), &request).unwrap();
        let stream = m.speak(request).await.unwrap();
        let (bytes, usage) = collect(encode_stream(
            stream,
            AudioEncoding::Pcm,
            EchoSpeechModel::SAMPLE_RATE,
            false,
        ))
        .await
        .unwrap();
        // Two words of 0.1 s at double speed, 2 bytes a sample.
        assert_eq!(bytes.len(), 2 * 2400 * 2);
        assert_eq!(usage.input_tokens, 2);
    }
}
