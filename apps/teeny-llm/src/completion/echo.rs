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

//! [`EchoCompletionModel`], the reference [`CompletionModel`].

use futures::future::BoxFuture;

use super::{
    CompletionCapabilities, CompletionEvent, CompletionFinishReason, CompletionLogprob,
    CompletionModel, CompletionPrompt, CompletionRequest, CompletionStream,
};
use crate::chat::SamplingSupport;
use crate::{ModelInfo, Result, Usage};

/// A completion model without weights: it "continues" each prompt by repeating it, word by
/// word, as the chat echo model repeats the user's message.
///
/// It honours `n`, `echo`, `stop`, `max_tokens` and `logprobs` (each word certain, log
/// probability 0, and its own only alternative), and takes token-id prompts, which it
/// writes as the ids. Tokens are whitespace-separated words.
#[derive(Debug, Clone, Copy, Default)]
pub struct EchoCompletionModel;

impl EchoCompletionModel {
    pub const MODEL_ID: &'static str = "teeny-echo-complete";
}

fn count(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

impl CompletionModel for EchoCompletionModel {
    fn info(&self) -> ModelInfo {
        ModelInfo {
            id: Self::MODEL_ID.to_owned(),
            created: 0,
            owned_by: "teenygrad".to_owned(),
        }
    }

    fn capabilities(&self) -> CompletionCapabilities {
        CompletionCapabilities {
            context_length: 4096,
            fill_in_middle: false,
            echo: true,
            max_best_of: 1,
            max_choices: 4,
            max_logprobs: Some(super::MAX_LOGPROBS),
            sampling: SamplingSupport::default(),
            stop_strings: true,
            token_prompts: true,
        }
    }

    fn complete(&self, request: CompletionRequest) -> BoxFuture<'_, Result<CompletionStream>> {
        Box::pin(async move {
            let mut events = Vec::new();
            let mut usage = Usage::default();
            let limit = request.max_tokens.map_or(usize::MAX, |n| n as usize);
            for (p, prompt) in request.prompts.iter().enumerate() {
                let text = match prompt {
                    CompletionPrompt::Text(t) => t.clone(),
                    CompletionPrompt::Tokens(ids) => {
                        ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
                    }
                };
                let words: Vec<&str> = text.split_whitespace().collect();
                usage.prompt_tokens = usage.prompt_tokens.saturating_add(count(words.len()));
                // The continuation stops before the first stop string.
                let cut = request
                    .sampling
                    .stop
                    .iter()
                    .filter_map(|s| text.find(s.as_str()))
                    .min()
                    .unwrap_or(text.len());
                let kept: Vec<&str> = text[..cut].split_whitespace().collect();
                let (emitted, reason) = if kept.len() > limit {
                    (&kept[..limit], CompletionFinishReason::Length)
                } else {
                    (&kept[..], CompletionFinishReason::Stop)
                };
                for j in 0..request.n {
                    let index = count(p) * request.n + j;
                    let mut offset = 0usize;
                    if request.echo {
                        events.push(CompletionEvent::Text {
                            index,
                            text: text.clone(),
                            logprobs: Vec::new(),
                        });
                        offset = text.len();
                    }
                    for (w, word) in emitted.iter().enumerate() {
                        let piece = if w == 0 && !request.echo {
                            (*word).to_owned()
                        } else {
                            format!(" {word}")
                        };
                        let logprobs = request
                            .logprobs
                            .map(|top| {
                                vec![CompletionLogprob {
                                    token: piece.clone(),
                                    logprob: 0.0,
                                    text_offset: count(offset),
                                    top: (top > 0)
                                        .then(|| (piece.clone(), 0.0))
                                        .into_iter()
                                        .collect(),
                                }]
                            })
                            .unwrap_or_default();
                        offset += piece.len();
                        events.push(CompletionEvent::Text {
                            index,
                            text: piece,
                            logprobs,
                        });
                    }
                    usage.completion_tokens =
                        usage.completion_tokens.saturating_add(count(emitted.len()));
                    events.push(CompletionEvent::Finish { index, reason });
                }
            }
            events.push(CompletionEvent::Usage(usage));
            let stream: CompletionStream =
                Box::pin(futures::stream::iter(events.into_iter().map(Ok)));
            Ok(stream)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::Sampling;
    use crate::completion::{collect, validate};

    async fn run(request: CompletionRequest) -> super::super::Completion {
        let m = EchoCompletionModel;
        validate(EchoCompletionModel::MODEL_ID, &m.capabilities(), &request).unwrap();
        let n = request.prompts.len() * request.n as usize;
        collect(n, m.complete(request).await.unwrap())
            .await
            .unwrap()
    }

    fn text(s: &str) -> CompletionPrompt {
        CompletionPrompt::Text(s.into())
    }

    #[tokio::test]
    async fn test_echo_completes_each_prompt_n_times_in_prompt_major_order() {
        let c = run(CompletionRequest {
            prompts: vec![text("a b"), text("c")],
            n: 2,
            ..CompletionRequest::default()
        })
        .await;
        let texts: Vec<&str> = c.choices.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, ["a b", "a b", "c", "c"]);
        assert_eq!(
            c.usage,
            Usage {
                prompt_tokens: 3,
                completion_tokens: 6
            }
        );
    }

    #[tokio::test]
    async fn test_echo_honours_echo_stop_limit_and_logprobs() {
        let c = run(CompletionRequest {
            prompts: vec![text("one two STOP three")],
            echo: true,
            logprobs: Some(1),
            sampling: Sampling {
                stop: vec![" STOP".into()],
                ..Sampling::default()
            },
            ..CompletionRequest::default()
        })
        .await;
        let ch = &c.choices[0];
        assert_eq!(ch.text, "one two STOP three one two");
        assert_eq!(ch.logprobs.len(), 2);
        assert_eq!(ch.logprobs[0].text_offset, 18);
        assert_eq!(ch.logprobs[1].top, vec![(" two".to_owned(), 0.0)]);
        let short = run(CompletionRequest {
            prompts: vec![CompletionPrompt::Tokens(vec![7, 8, 9])],
            max_tokens: Some(2),
            ..CompletionRequest::default()
        })
        .await;
        assert_eq!(short.choices[0].text, "7 8");
        assert_eq!(
            short.choices[0].finish_reason,
            CompletionFinishReason::Length
        );
    }
}
