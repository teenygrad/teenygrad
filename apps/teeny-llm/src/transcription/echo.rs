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

//! [`EchoTranscriptionModel`], the reference [`TranscriptionModel`].

use futures::future::BoxFuture;

use super::{
    Segment, Timestamps, Transcription, TranscriptionCapabilities, TranscriptionEvent,
    TranscriptionModel, TranscriptionRequest, TranscriptionStream, TranscriptionTask,
    TranscriptionUsage, Word,
};
use crate::audio::Pcm;
use crate::{ModelInfo, Result};

/// A transcription model without weights: it hears no words, and "transcribes" a
/// recording as a description of it (`"1.5 seconds of audio"`, `"... in English"` when
/// translating), one word at a time, with a segment and word timings spread evenly over
/// the recording.
#[derive(Debug, Clone, Copy, Default)]
pub struct EchoTranscriptionModel;

impl EchoTranscriptionModel {
    pub const MODEL_ID: &'static str = "teeny-echo-transcribe";
}

impl TranscriptionModel for EchoTranscriptionModel {
    fn info(&self) -> ModelInfo {
        ModelInfo {
            id: Self::MODEL_ID.to_owned(),
            created: 0,
            owned_by: "teenygrad".to_owned(),
        }
    }

    fn capabilities(&self) -> TranscriptionCapabilities {
        TranscriptionCapabilities {
            languages: Vec::new(),
            translation: true,
            timestamps: Timestamps {
                segments: true,
                words: true,
            },
            streaming: true,
            logprobs: true,
            max_seconds: 600.0,
            sample_rate: 16_000,
        }
    }

    fn transcribe(
        &self,
        request: TranscriptionRequest,
        audio: Pcm,
    ) -> BoxFuture<'_, Result<TranscriptionStream>> {
        Box::pin(async move {
            let duration = audio.duration();
            let mut text = format!("{duration:.1} seconds of audio");
            if request.task == TranscriptionTask::Translate {
                text.push_str(" in English");
            }
            let words: Vec<&str> = text.split(' ').collect();
            let step = duration / words.len() as f32;
            let mut events: Vec<TranscriptionEvent> = words
                .iter()
                .enumerate()
                .map(|(i, w)| {
                    TranscriptionEvent::Delta(if i == 0 {
                        (*w).to_owned()
                    } else {
                        format!(" {w}")
                    })
                })
                .collect();
            let n = u32::try_from(words.len()).unwrap_or(u32::MAX);
            let done = Transcription {
                language: Some(request.language.clone().unwrap_or_else(|| "en".into())),
                duration,
                segments: if request.timestamps.segments {
                    vec![Segment {
                        id: 0,
                        start: 0.0,
                        end: duration,
                        text: text.clone(),
                    }]
                } else {
                    Vec::new()
                },
                words: if request.timestamps.words {
                    words
                        .iter()
                        .enumerate()
                        .map(|(i, w)| Word {
                            word: (*w).to_owned(),
                            start: i as f32 * step,
                            end: (i + 1) as f32 * step,
                        })
                        .collect()
                } else {
                    Vec::new()
                },
                logprobs: if request.logprobs {
                    words.iter().map(|w| ((*w).to_owned(), 0.0)).collect()
                } else {
                    Vec::new()
                },
                usage: TranscriptionUsage {
                    seconds: duration,
                    input_tokens: 0,
                    output_tokens: n,
                },
                text,
            };
            events.push(TranscriptionEvent::Done(done));
            let stream: TranscriptionStream =
                Box::pin(futures::stream::iter(events.into_iter().map(Ok)));
            Ok(stream)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transcription::collect;

    #[tokio::test]
    async fn test_echo_transcribes_a_description_with_timings() {
        let audio = Pcm {
            sample_rate: 16_000,
            channels: 1,
            samples: vec![0.0; 24_000],
        };
        let request = TranscriptionRequest {
            task: TranscriptionTask::Translate,
            timestamps: Timestamps {
                segments: true,
                words: true,
            },
            ..TranscriptionRequest::default()
        };
        let t = collect(
            EchoTranscriptionModel
                .transcribe(request, audio)
                .await
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(t.text, "1.5 seconds of audio in English");
        assert_eq!(t.segments[0].end, 1.5);
        assert_eq!(t.words.len(), 6);
        assert!((t.words[5].end - 1.5).abs() < 1e-6);
        assert!(t.logprobs.is_empty());
    }
}
