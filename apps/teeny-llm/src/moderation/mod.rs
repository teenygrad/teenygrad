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

//! Content classification (`/v1/moderations`), useful in front of a public server.
//!
//! A model implements [`ModerationModel`]: it declares its [`ModerationCapabilities`]
//! (the [`ModerationCategory`]s it scores, and the input it reads) and classifies each
//! input of a [`ModerationRequest`]; [`validate`] refuses what it cannot do first.
//! [`EchoModerationModel`] is the reference implementation.

mod echo;

pub use echo::EchoModerationModel;

use futures::future::BoxFuture;

use crate::{Error, ModelInfo, Result};

/// A kind of harmful content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ModerationCategory {
    Harassment,
    HarassmentThreatening,
    Hate,
    HateThreatening,
    Illicit,
    IllicitViolent,
    SelfHarm,
    SelfHarmIntent,
    SelfHarmInstructions,
    Sexual,
    SexualMinors,
    Violence,
    ViolenceGraphic,
}

impl ModerationCategory {
    /// Every category, in the order the API lists them.
    pub const ALL: [Self; 13] = [
        Self::Harassment,
        Self::HarassmentThreatening,
        Self::Hate,
        Self::HateThreatening,
        Self::Illicit,
        Self::IllicitViolent,
        Self::SelfHarm,
        Self::SelfHarmIntent,
        Self::SelfHarmInstructions,
        Self::Sexual,
        Self::SexualMinors,
        Self::Violence,
        Self::ViolenceGraphic,
    ];

    /// Its name in the API (`"self-harm/intent"`).
    pub fn name(self) -> &'static str {
        match self {
            Self::Harassment => "harassment",
            Self::HarassmentThreatening => "harassment/threatening",
            Self::Hate => "hate",
            Self::HateThreatening => "hate/threatening",
            Self::Illicit => "illicit",
            Self::IllicitViolent => "illicit/violent",
            Self::SelfHarm => "self-harm",
            Self::SelfHarmIntent => "self-harm/intent",
            Self::SelfHarmInstructions => "self-harm/instructions",
            Self::Sexual => "sexual",
            Self::SexualMinors => "sexual/minors",
            Self::Violence => "violence",
            Self::ViolenceGraphic => "violence/graphic",
        }
    }
}

/// What an input is made of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    Text,
    Image,
}

impl Modality {
    pub fn name(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
        }
    }
}

/// A piece of an input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModerationPart {
    Text(String),
    /// An image, by URL (or `data:` URL).
    Image(String),
}

impl ModerationPart {
    pub fn modality(&self) -> Modality {
        match self {
            Self::Text(_) => Modality::Text,
            Self::Image(_) => Modality::Image,
        }
    }
}

/// A request to classify inputs, each one or more parts (text and images together are one
/// input).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModerationRequest {
    /// The model; `None` leaves the choice to the engine.
    pub model: Option<String>,
    pub inputs: Vec<Vec<ModerationPart>>,
}

/// How one input scored in one category.
#[derive(Debug, Clone, PartialEq)]
pub struct CategoryScore {
    pub category: ModerationCategory,
    pub flagged: bool,
    /// 0 to 1.
    pub score: f32,
    /// The kinds of part the score came from.
    pub applied: Vec<Modality>,
}

/// How one input was classified.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModerationResult {
    /// Whether any category flagged it.
    pub flagged: bool,
    /// The categories the model scored; those it does not are absent.
    pub categories: Vec<CategoryScore>,
}

impl ModerationResult {
    /// A result over `categories`, flagged when any of them is.
    pub fn new(categories: Vec<CategoryScore>) -> Self {
        Self {
            flagged: categories.iter().any(|c| c.flagged),
            categories,
        }
    }

    /// The score for `category`, if the model gave one.
    pub fn get(&self, category: ModerationCategory) -> Option<&CategoryScore> {
        self.categories.iter().find(|c| c.category == category)
    }
}

/// The classification of every input, in order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Moderation {
    /// The model that classified them.
    pub model: String,
    pub results: Vec<ModerationResult>,
}

/// What a moderation model can do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModerationCapabilities {
    pub categories: Vec<ModerationCategory>,
    pub images: bool,
    /// The longest text of an input, in characters.
    pub max_input_chars: usize,
    /// The most inputs in one request.
    pub max_batch: usize,
}

impl Default for ModerationCapabilities {
    fn default() -> Self {
        Self {
            categories: ModerationCategory::ALL.to_vec(),
            images: false,
            max_input_chars: 32_768,
            max_batch: 32,
        }
    }
}

fn invalid(detail: impl Into<String>) -> Error {
    Error::InvalidParameter {
        parameter: "input".into(),
        detail: detail.into(),
    }
}

/// Checks `request` for the model `model` with capabilities `caps`.
pub fn validate(model: &str, caps: &ModerationCapabilities, r: &ModerationRequest) -> Result<()> {
    if r.inputs.is_empty() {
        return Err(invalid("there is nothing to classify"));
    }
    if r.inputs.len() > caps.max_batch {
        return Err(invalid(format!(
            "{} inputs; the model takes at most {} at once",
            r.inputs.len(),
            caps.max_batch
        )));
    }
    for (i, parts) in r.inputs.iter().enumerate() {
        if parts.is_empty() {
            return Err(invalid(format!("input {i} has no parts")));
        }
        let chars: usize = parts
            .iter()
            .map(|p| match p {
                ModerationPart::Text(t) => t.chars().count(),
                ModerationPart::Image(_) => 0,
            })
            .sum();
        if chars > caps.max_input_chars {
            return Err(invalid(format!(
                "input {i} has {chars} characters; the model reads at most {}",
                caps.max_input_chars
            )));
        }
        if !caps.images && parts.iter().any(|p| p.modality() == Modality::Image) {
            return Err(Error::UnsupportedParameter {
                model: model.to_owned(),
                parameter: "input".into(),
                detail: "images (it classifies text only)".into(),
            });
        }
    }
    Ok(())
}

/// A model that classifies content.
pub trait ModerationModel: Send + Sync + 'static {
    fn info(&self) -> ModelInfo;

    fn capabilities(&self) -> ModerationCapabilities;

    /// Classifies each input of `request`, which has passed [`validate`], in order.
    fn moderate(&self, request: ModerationRequest) -> BoxFuture<'_, Result<Moderation>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(s: &str) -> Vec<ModerationPart> {
        vec![ModerationPart::Text(s.into())]
    }

    #[test]
    fn test_validate_refuses_what_the_model_cannot_do() {
        let caps = ModerationCapabilities {
            max_input_chars: 4,
            max_batch: 2,
            ..ModerationCapabilities::default()
        };
        let ok = ModerationRequest {
            model: None,
            inputs: vec![text("abcd")],
        };
        assert!(validate("m", &caps, &ok).is_ok());
        let bad = [
            vec![],
            vec![text("a"), text("b"), text("c")],
            vec![vec![]],
            vec![text("abcde")],
            vec![vec![ModerationPart::Image("data:,".into())]],
        ];
        for inputs in bad {
            let r = validate(
                "m",
                &caps,
                &ModerationRequest {
                    model: None,
                    inputs,
                },
            );
            assert!(
                matches!(
                    r,
                    Err(Error::InvalidParameter { ref parameter, .. } | Error::UnsupportedParameter { ref parameter, .. })
                        if parameter == "input"
                ),
                "{r:?}"
            );
        }
    }

    #[test]
    fn test_category_names_follow_the_api() {
        let names: Vec<&str> = ModerationCategory::ALL.iter().map(|c| c.name()).collect();
        assert_eq!(
            names[6..9],
            ["self-harm", "self-harm/intent", "self-harm/instructions"]
        );
        assert_eq!(names.len(), 13);
    }
}
