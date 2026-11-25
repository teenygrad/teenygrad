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

use crate::{
    Engine, Error, FinishReason, GenerationChunk, GenerationRequest, ModelInfo, Prompt, Result,
    Role, TokenStream, Usage,
};

/// Placeholder engine that streams back the prompt's last user message word by word.
///
/// It lets the server and CLI be exercised end to end before real model execution exists. Token
/// counts are whitespace-separated words, which is only a rough stand-in for a tokenizer.
#[derive(Debug, Clone)]
pub struct EchoEngine {
    model: ModelInfo,
}

impl EchoEngine {
    pub const MODEL_ID: &'static str = "teeny-echo";

    pub fn new() -> Self {
        Self {
            model: ModelInfo {
                id: Self::MODEL_ID.to_owned(),
                created: 0,
                owned_by: "teenygrad".to_owned(),
            },
        }
    }
}

impl Default for EchoEngine {
    fn default() -> Self {
        Self::new()
    }
}

fn word_count(text: &str) -> u32 {
    u32::try_from(text.split_whitespace().count()).unwrap_or(u32::MAX)
}

impl Engine for EchoEngine {
    fn name(&self) -> &'static str {
        "echo"
    }

    fn models(&self) -> Vec<ModelInfo> {
        vec![self.model.clone()]
    }

    fn generate_stream(&self, request: GenerationRequest) -> Result<TokenStream> {
        self.model(&request.model)?;

        let (prompt_tokens, reply) = match &request.prompt {
            Prompt::Chat(messages) => {
                let reply = messages
                    .iter()
                    .rev()
                    .find(|m| m.role == Role::User)
                    .map(|m| m.content.clone())
                    .ok_or_else(|| {
                        Error::InvalidRequest(format!(
                            "chat prompt has {} message(s) but none from the user",
                            messages.len()
                        ))
                    })?;
                let prompt_tokens = messages.iter().map(|m| word_count(&m.content)).sum();
                (prompt_tokens, reply)
            }
            Prompt::Text(text) => (word_count(text), text.clone()),
        };

        let max_tokens = request
            .sampling
            .max_tokens
            .map_or(usize::MAX, |n| n as usize);
        let words: Vec<&str> = reply.split_whitespace().collect();
        let (emitted, finish_reason) = if words.len() > max_tokens {
            (&words[..max_tokens], FinishReason::Length)
        } else {
            (&words[..], FinishReason::Stop)
        };

        let mut chunks: Vec<Result<GenerationChunk>> = emitted
            .iter()
            .enumerate()
            .map(|(i, w)| {
                Ok(GenerationChunk {
                    text: if i == 0 {
                        (*w).to_owned()
                    } else {
                        format!(" {w}")
                    },
                    finish_reason: None,
                    usage: None,
                })
            })
            .collect();
        chunks.push(Ok(GenerationChunk {
            text: String::new(),
            finish_reason: Some(finish_reason),
            usage: Some(Usage {
                prompt_tokens,
                completion_tokens: u32::try_from(emitted.len()).unwrap_or(u32::MAX),
            }),
        }));

        Ok(Box::pin(futures::stream::iter(chunks)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChatMessage, SamplingParams};

    fn request(prompt: Prompt, max_tokens: Option<u32>) -> GenerationRequest {
        GenerationRequest {
            model: EchoEngine::MODEL_ID.to_owned(),
            prompt,
            sampling: SamplingParams {
                max_tokens,
                ..SamplingParams::default()
            },
        }
    }

    #[tokio::test]
    async fn test_echoes_last_user_message() {
        let engine: &dyn Engine = &EchoEngine::new();
        let prompt = Prompt::Chat(vec![
            ChatMessage::system("be brief"),
            ChatMessage::user("hello there"),
        ]);
        let out = engine.generate(request(prompt, None)).await.unwrap();
        assert_eq!(out.text, "hello there");
        assert_eq!(out.finish_reason, FinishReason::Stop);
        assert_eq!(out.usage.prompt_tokens, 4);
        assert_eq!(out.usage.completion_tokens, 2);
    }

    #[tokio::test]
    async fn test_truncates_at_max_tokens() {
        let engine: &dyn Engine = &EchoEngine::new();
        let out = engine
            .generate(request(Prompt::Text("a b c d".into()), Some(2)))
            .await
            .unwrap();
        assert_eq!(out.text, "a b");
        assert_eq!(out.finish_reason, FinishReason::Length);
    }

    #[test]
    fn test_rejects_unknown_model() {
        let engine = EchoEngine::new();
        let mut req = request(Prompt::Text("x".into()), None);
        req.model = "gpt-4o".into();
        assert!(matches!(
            engine.generate_stream(req),
            Err(Error::ModelNotFound { .. })
        ));
    }
}
