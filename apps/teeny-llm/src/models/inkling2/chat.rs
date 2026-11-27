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

//! Prompts and replies for Inkling-2: the checkpoint's own `chat_template.jinja`
//! rendered with minijinja, and its `tokenizer.json`.
//!
//! Two things differ from the Gemma template and are easy to get wrong:
//!
//! - **There is no generation prompt.** The template never mentions
//!   `add_generation_prompt`; it ends the last turn with `<|end_message|>` and the
//!   model emits `<|message_model|>` itself. Appending one here would put a token in
//!   the prompt that the training data never had in that position.
//! - **It is a thinking model.** The template injects a system turn reading
//!   `Thinking effort level: <x>` from `reasoning_effort`, defaulting to 0.9. The
//!   default is the template's, not ours, so leaving it unset matches Hugging Face.
//!
//! The template also calls Jinja's `raise_exception`, which minijinja does not provide
//! as a builtin, so it is registered below.

use std::path::Path;

use minijinja::{Environment, Error, ErrorKind, context};
use serde_json::Value;
use tokenizers::Tokenizer;

/// The tokenizer and chat template of the Inkling-2 checkpoint folder.
pub struct Chat {
    tokenizer: Tokenizer,
    env: Environment<'static>,
    /// The id generation stops on (`<|content_model_end_sampling|>`).
    pub eos: u32,
    /// `<|message_model|>`, which opens the model's turn.
    pub turn_start: u32,
}

impl Chat {
    /// Loads `tokenizer.json`, `chat_template.jinja` and `special_tokens_map.json`
    /// from `dir`.
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("loading {}/tokenizer.json: {e}", dir.display()))?;
        let template = std::fs::read_to_string(dir.join("chat_template.jinja"))
            .map_err(|e| anyhow::anyhow!("reading {}/chat_template.jinja: {e}", dir.display()))?;
        // `{% generation %}`/`{% endgeneration %}` is a Hugging Face extension that marks
        // which span of the render is assistant-generated, so a trainer can build a loss
        // mask for it. It emits nothing, and minijinja has no such statement, so the
        // markers come out rather than the renderer failing on them.
        let template = template
            .replace("{%- generation -%}", "")
            .replace("{% generation %}", "")
            .replace("{%- generation %}", "")
            .replace("{% generation -%}", "")
            .replace("{%- endgeneration -%}", "")
            .replace("{% endgeneration %}", "")
            .replace("{%- endgeneration %}", "")
            .replace("{% endgeneration -%}", "");
        let special: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("special_tokens_map.json")).map_err(|e| {
                anyhow::anyhow!("reading {}/special_tokens_map.json: {e}", dir.display())
            })?,
        )?;
        let eos_token = match &special["eos_token"] {
            Value::String(s) => s.clone(),
            v => v["content"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("special_tokens_map.json has no eos_token"))?
                .to_string(),
        };
        let eos = tokenizer
            .token_to_id(&eos_token)
            .ok_or_else(|| anyhow::anyhow!("the tokenizer has no id for {eos_token}"))?;

        let mut env = Environment::new();
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        // Hugging Face renders chat templates with this in scope; the Inkling-2
        // template calls it when `reasoning_effort` is not one it knows.
        env.add_function("raise_exception", |msg: String| -> Result<(), Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_template_owned("chat", template)?;
        let turn_start = tokenizer
            .token_to_id("<|message_model|>")
            .ok_or_else(|| anyhow::anyhow!("the tokenizer has no <|message_model|>"))?;
        Ok(Self {
            tokenizer,
            env,
            eos,
            turn_start,
        })
    }

    /// Renders `messages` the way `apply_chat_template` does.
    ///
    /// `reasoning_effort` is passed through when given: a name the template knows
    /// (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`) or a number. When
    /// `None` the template applies its own default, as Hugging Face does.
    pub fn render(
        &self,
        messages: &Value,
        reasoning_effort: Option<&str>,
    ) -> anyhow::Result<String> {
        Ok(self.env.get_template("chat")?.render(context! {
            messages => messages,
            reasoning_effort => reasoning_effort,
        })?)
    }

    /// The token ids of `messages`.
    ///
    /// The template writes every special token itself and stops after the user's
    /// `<|end_message|>`; the model opens its own turn with `<|message_model|>`, as
    /// the reference's own `generate` does, so nothing is appended here.
    pub fn prompt(
        &self,
        messages: &Value,
        reasoning_effort: Option<&str>,
    ) -> anyhow::Result<Vec<u32>> {
        self.encode(&self.render(messages, reasoning_effort)?)
    }

    /// As [`Self::prompt`], without the token that opens the model's turn.
    pub fn prompt_raw(
        &self,
        messages: &Value,
        reasoning_effort: Option<&str>,
    ) -> anyhow::Result<Vec<u32>> {
        self.encode(&self.render(messages, reasoning_effort)?)
    }

    /// Tokenizes `text` with no added special tokens.
    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decodes `ids`, keeping the special tokens so the structure stays visible.
    pub fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.tokenizer
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!("decoding: {e}"))
    }

    /// Decodes `ids` as readable text, dropping the special tokens.
    pub fn decode_visible(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("decoding: {e}"))
    }
}
