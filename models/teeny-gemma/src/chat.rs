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

//! Prompts and replies for Gemma chat models: the model's own `chat_template.jinja`
//! rendered with minijinja (Python-compatible methods enabled, as Hugging Face renders
//! it with Jinja2), its `tokenizer.json`, and FunctionGemma's function-call syntax.

use std::path::Path;

use minijinja::{Environment, context};
use serde_json::Value;
use tokenizers::Tokenizer;

/// The tokenizer and chat template of a Hugging Face model folder.
pub struct Chat {
    tokenizer: Tokenizer,
    env: Environment<'static>,
    bos: String,
}

impl Chat {
    /// Loads `tokenizer.json`, `chat_template.jinja` and the BOS token from
    /// `special_tokens_map.json` in `dir`.
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("loading {}/tokenizer.json: {e}", dir.display()))?;
        let template = std::fs::read_to_string(dir.join("chat_template.jinja"))
            .map_err(|e| anyhow::anyhow!("reading {}/chat_template.jinja: {e}", dir.display()))?;
        let special: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("special_tokens_map.json")).map_err(|e| {
                anyhow::anyhow!("reading {}/special_tokens_map.json: {e}", dir.display())
            })?,
        )?;
        let bos = match &special["bos_token"] {
            Value::String(s) => s.clone(),
            v => v["content"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("special_tokens_map.json has no bos_token"))?
                .to_string(),
        };
        let mut env = Environment::new();
        minijinja_contrib::add_to_environment(&mut env);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template_owned("chat", template)?;
        Ok(Self {
            tokenizer,
            env,
            bos,
        })
    }

    /// Renders `messages` (and `tools`, OpenAI-style JSON) with the generation prompt.
    pub fn render(&self, messages: &Value, tools: &Value) -> anyhow::Result<String> {
        let tools = if tools.as_array().is_some_and(|a| !a.is_empty()) {
            Some(tools)
        } else {
            None
        };
        Ok(self.env.get_template("chat")?.render(context! {
            messages => messages,
            tools => tools,
            bos_token => &self.bos,
            add_generation_prompt => true,
        })?)
    }

    /// The token ids of `messages` and `tools`, as Hugging Face's
    /// `apply_chat_template(..., add_generation_prompt=True)` produces them.
    pub fn prompt(&self, messages: &Value, tools: &Value) -> anyhow::Result<Vec<u32>> {
        let text = self.render(messages, tools)?;
        // The template writes the BOS token itself, so no special tokens are added.
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenizing prompt: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Decodes `ids`, keeping special tokens (FunctionGemma's call markers are special).
    pub fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.tokenizer
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!("decoding: {e}"))
    }
}

/// A function call parsed from FunctionGemma's output.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionCall {
    /// The function name, empty when the model left it out. See [`parse_calls`].
    pub name: String,
    /// The arguments, as JSON.
    pub args: Value,
}

/// Parses every `<start_function_call>call:NAME{ARGS}<end_function_call>` in `text`.
/// Arguments use FunctionGemma's syntax: `key:value` pairs, strings wrapped in
/// `<escape>`, nested `{...}` objects and `[...]` lists.
///
/// The name is sometimes missing. Asked "tell me the temperature in Lisbon", the 270M model
/// emits `call{city:<escape>Lisbon<escape>}`: it has decided to call, and with the right
/// argument, but left out which function. That parses here with an empty [`FunctionCall::name`]
/// rather than being thrown away, because a caller that declared one function knows what was
/// meant. Refusing it loses the turn, and the visitor sees nothing at all.
pub fn parse_calls(text: &str) -> anyhow::Result<Vec<FunctionCall>> {
    const START: &str = "<start_function_call>call";
    const END: &str = "<end_function_call>";
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find(START) {
        let body = &rest[i + START.len()..];
        let j = body
            .find(END)
            .ok_or_else(|| anyhow::anyhow!("unterminated function call: {body:?}"))?;
        let call = &body[..j];
        let brace = call
            .find('{')
            .ok_or_else(|| anyhow::anyhow!("function call without arguments: {call:?}"))?;
        // `call:name{...}` and `call{...}` both arrive here; the colon is not part of a name.
        let name = call[..brace].trim().trim_start_matches(':').trim();
        let mut p = Parser {
            s: &call[brace..],
            pos: 0,
        };
        let args = p.value()?;
        calls.push(FunctionCall {
            name: name.to_string(),
            args,
        });
        rest = &body[j + END.len()..];
    }
    Ok(calls)
}

struct Parser<'a> {
    s: &'a str,
    pos: usize,
}

impl Parser<'_> {
    const ESC: &'static str = "<escape>";

    fn rest(&self) -> &str {
        &self.s[self.pos..]
    }

    fn eat(&mut self, c: char) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.rest().starts_with(c),
            "expected {c:?} at {:?}",
            self.rest()
        );
        self.pos += c.len_utf8();
        Ok(())
    }

    fn value(&mut self) -> anyhow::Result<Value> {
        // The model puts a space after the colon often enough to matter, and without this the
        // value falls through to the bare-scalar arm and the `<escape>` markers end up inside
        // the string.
        let skip = self.rest().len() - self.rest().trim_start().len();
        self.pos += skip;
        let r = self.rest();
        if r.starts_with(Self::ESC) {
            self.pos += Self::ESC.len();
            let end = self
                .rest()
                .find(Self::ESC)
                .ok_or_else(|| anyhow::anyhow!("unterminated <escape> string"))?;
            let s = self.rest()[..end].to_string();
            self.pos += end + Self::ESC.len();
            Ok(Value::String(s))
        } else if r.starts_with('{') {
            self.eat('{')?;
            let mut map = serde_json::Map::new();
            while !self.rest().starts_with('}') {
                let colon = self
                    .rest()
                    .find(':')
                    .ok_or_else(|| anyhow::anyhow!("key without value at {:?}", self.rest()))?;
                // Keys pick up the space after a comma the same way values pick up the one
                // after a colon.
                let key = self.rest()[..colon]
                    .trim()
                    .trim_start_matches(Self::ESC)
                    .trim_end_matches(Self::ESC)
                    .to_string();
                self.pos += colon + 1;
                map.insert(key, self.value()?);
                if self.rest().starts_with(',') {
                    self.eat(',')?;
                }
            }
            self.eat('}')?;
            Ok(Value::Object(map))
        } else if r.starts_with('[') {
            self.eat('[')?;
            let mut items = Vec::new();
            while !self.rest().starts_with(']') {
                items.push(self.value()?);
                if self.rest().starts_with(',') {
                    self.eat(',')?;
                }
            }
            self.eat(']')?;
            Ok(Value::Array(items))
        } else {
            // A bare scalar: number, true, false or null, up to the next delimiter.
            let end = r.find([',', '}', ']']).unwrap_or(r.len());
            let tok = r[..end].trim().to_string();
            self.pos += end;
            Ok(serde_json::from_str(&tok).unwrap_or(Value::String(tok)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_function_calls() {
        let text = "<start_function_call>call:get_weather{city:<escape>Paris<escape>,days:3,\
                    opts:{metric:true},tags:[<escape>a<escape>,<escape>b<escape>]}\
                    <end_function_call><start_function_response>";
        let calls = parse_calls(text).unwrap();
        assert_eq!(
            calls,
            vec![FunctionCall {
                name: "get_weather".into(),
                args: serde_json::json!({
                    "city": "Paris", "days": 3, "opts": {"metric": true}, "tags": ["a", "b"]
                }),
            }]
        );
    }

    #[test]
    fn parses_values_written_with_a_space_after_the_colon() {
        let text = "<start_function_call>call:get_weather{city: <escape>Ibarreta<escape>, \
                    days: 3}<end_function_call>";
        let calls = parse_calls(text).unwrap();
        assert_eq!(calls[0].args["city"], "Ibarreta");
        assert_eq!(calls[0].args["days"], 3);
    }

    #[test]
    fn parses_a_call_that_left_out_its_name() {
        // Exactly what the board emitted for "Tell me the temperature in Lisbon."
        let text = "<start_function_call>call{city:<escape>Lisbon<escape>}\
                    <end_function_call><start_function_response>";
        let calls = parse_calls(text).unwrap();
        assert_eq!(
            calls,
            vec![FunctionCall {
                name: String::new(),
                args: serde_json::json!({"city": "Lisbon"}),
            }]
        );
    }

    #[test]
    fn parses_both_forms_in_one_reply() {
        let text = "<start_function_call>call:get_weather{city:<escape>Oslo<escape>}\
                    <end_function_call>\
                    <start_function_call>call{city:<escape>Lima<escape>}<end_function_call>";
        let calls = parse_calls(text).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[1].name, "");
        assert_eq!(calls[1].args["city"], "Lima");
    }

    #[test]
    fn still_refuses_a_call_with_no_arguments() {
        let text = "<start_function_call>call:get_weather<end_function_call>";
        assert!(parse_calls(text).is_err());
    }
}
