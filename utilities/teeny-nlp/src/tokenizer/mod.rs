/*
 * Copyright (c) 2026 Teenygrad.
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

//! Tokenizer trait and chat-message types.

use serde::{Deserialize, Serialize};

/// A single chat-conversation turn: a `role` (e.g. `"user"`, `"assistant"`, `"system"`) paired
/// with its `content`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Who sent this turn (e.g. `"user"`, `"assistant"`, `"system"`).
    pub role: String,
    /// The turn's text content.
    pub content: String,
}

impl Message {
    /// Creates a message from `role`/`content` string slices.
    pub fn new(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: content.to_string(),
        }
    }
}

/// Text tokenization and chat-templating, implemented per model/tokenizer family.
pub trait Tokenizer {
    /// Renders `messages` through `chat_template` (a Jinja-style template string, per the Hugging
    /// Face chat-template convention), optionally tokenizing the result (`tokenize`), appending a
    /// generation prompt (`add_generation_prompt`), and enabling "thinking"/reasoning mode
    /// (`enable_thinking`) for models that support it.
    fn apply_chat_template(
        &self,
        messages: &[Message],
        chat_template: &str,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> String;

    /// Encodes `texts` into a flat sequence of token IDs.
    fn encode(&self, texts: &[String]) -> Vec<usize>;

    /// Decodes a sequence of token IDs back into text.
    fn decode(&self, ids: &[usize]) -> String;
}
