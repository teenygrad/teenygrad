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

//! `/v1/moderations`: `CreateModerationRequest` and `CreateModerationResponse`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use teeny_llm::moderation::{
    Moderation, ModerationCategory, ModerationPart, ModerationRequest, ModerationResult,
};

/// An image, by URL.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct ImageUrl {
    pub url: String,
}

/// A part of a multimodal input.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

/// `input`: a text, several texts (one result each), or the parts of one multimodal
/// input.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum ModerationInput {
    One(String),
    Many(Vec<String>),
    Parts(Vec<InputPart>),
}

/// `CreateModerationRequest`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct CreateModerationRequest {
    pub input: ModerationInput,
    #[serde(default)]
    pub model: Option<String>,
}

impl CreateModerationRequest {
    pub fn into_core(self) -> ModerationRequest {
        let text = |s: String| vec![ModerationPart::Text(s)];
        let inputs = match self.input {
            ModerationInput::One(s) => vec![text(s)],
            ModerationInput::Many(v) => v.into_iter().map(text).collect(),
            ModerationInput::Parts(parts) => vec![
                parts
                    .into_iter()
                    .map(|p| match p {
                        InputPart::Text { text } => ModerationPart::Text(text),
                        InputPart::ImageUrl { image_url } => ModerationPart::Image(image_url.url),
                    })
                    .collect(),
            ],
        };
        ModerationRequest {
            model: self.model,
            inputs,
        }
    }
}

/// One input's classification. Every category is listed; those the model does not score
/// are unflagged, scored 0, and applied to nothing.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModerationResultJson {
    pub flagged: bool,
    pub categories: Map<String, Value>,
    pub category_scores: Map<String, Value>,
    pub category_applied_input_types: Map<String, Value>,
}

impl ModerationResultJson {
    pub fn from_core(r: &ModerationResult) -> Self {
        let mut categories = Map::new();
        let mut category_scores = Map::new();
        let mut applied = Map::new();
        for c in ModerationCategory::ALL {
            let name = c.name().to_owned();
            let score = r.get(c);
            categories.insert(name.clone(), Value::Bool(score.is_some_and(|s| s.flagged)));
            category_scores.insert(
                name.clone(),
                Value::from(score.map_or(0.0, |s| f64::from(s.score))),
            );
            applied.insert(
                name,
                Value::from(
                    score
                        .map(|s| s.applied.iter().map(|m| m.name()).collect::<Vec<_>>())
                        .unwrap_or_default(),
                ),
            );
        }
        Self {
            flagged: r.flagged,
            categories,
            category_scores,
            category_applied_input_types: applied,
        }
    }
}

/// `CreateModerationResponse`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CreateModerationResponse {
    pub id: String,
    pub model: String,
    pub results: Vec<ModerationResultJson>,
}

impl CreateModerationResponse {
    pub fn from_core(id: String, m: &Moderation) -> Self {
        Self {
            id,
            model: m.model.clone(),
            results: m
                .results
                .iter()
                .map(ModerationResultJson::from_core)
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use teeny_llm::moderation::{CategoryScore, Modality};

    fn parse(v: Value) -> ModerationRequest {
        serde_json::from_value::<CreateModerationRequest>(v)
            .unwrap()
            .into_core()
    }

    #[test]
    fn test_inputs_map_to_one_result_each() {
        assert_eq!(parse(serde_json::json!({"input": "a"})).inputs.len(), 1);
        let many = parse(serde_json::json!({"input": ["a", "b"], "model": "m"}));
        assert_eq!((many.inputs.len(), many.model.as_deref()), (2, Some("m")));
        let parts = parse(serde_json::json!({"input": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:,"}}
        ]}));
        assert_eq!(
            parts.inputs,
            vec![vec![
                ModerationPart::Text("look".into()),
                ModerationPart::Image("data:,".into())
            ]]
        );
    }

    #[test]
    fn test_results_list_every_category() {
        let r = ModerationResult::new(vec![CategoryScore {
            category: ModerationCategory::Hate,
            flagged: true,
            score: 0.75,
            applied: vec![Modality::Text],
        }]);
        let j = serde_json::to_value(ModerationResultJson::from_core(&r)).unwrap();
        assert_eq!(j["flagged"], true);
        assert_eq!(j["categories"].as_object().unwrap().len(), 13);
        assert_eq!(j["categories"]["hate"], true);
        assert_eq!(j["category_scores"]["hate"], 0.75);
        assert_eq!(j["category_scores"]["sexual/minors"], 0.0);
        assert_eq!(
            j["category_applied_input_types"]["hate"],
            serde_json::json!(["text"])
        );
    }
}
