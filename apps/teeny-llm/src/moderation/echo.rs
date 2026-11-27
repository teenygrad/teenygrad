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

//! [`EchoModerationModel`], the reference [`ModerationModel`].

use futures::future::BoxFuture;

use super::{
    CategoryScore, Modality, Moderation, ModerationCapabilities, ModerationCategory,
    ModerationModel, ModerationPart, ModerationRequest, ModerationResult,
};
use crate::{ModelInfo, Result};

/// A moderation model without weights: it judges nothing by meaning, and flags a category
/// only when the text names it as `flag:<category>` (`flag:violence`,
/// `flag:self-harm/intent`), scoring it 1 and every other category 0. It reads text only.
#[derive(Debug, Clone, Copy, Default)]
pub struct EchoModerationModel;

impl EchoModerationModel {
    pub const MODEL_ID: &'static str = "teeny-echo-moderate";
}

impl ModerationModel for EchoModerationModel {
    fn info(&self) -> ModelInfo {
        ModelInfo {
            id: Self::MODEL_ID.to_owned(),
            created: 0,
            owned_by: "teenygrad".to_owned(),
        }
    }

    fn capabilities(&self) -> ModerationCapabilities {
        ModerationCapabilities::default()
    }

    fn moderate(&self, request: ModerationRequest) -> BoxFuture<'_, Result<Moderation>> {
        Box::pin(async move {
            let results = request
                .inputs
                .iter()
                .map(|parts| {
                    let flags: Vec<&str> = parts
                        .iter()
                        .filter_map(|p| match p {
                            ModerationPart::Text(t) => Some(t.split_whitespace()),
                            ModerationPart::Image(_) => None,
                        })
                        .flatten()
                        .filter_map(|w| w.strip_prefix("flag:"))
                        .collect();
                    ModerationResult::new(
                        ModerationCategory::ALL
                            .iter()
                            .map(|&category| {
                                let flagged = flags.contains(&category.name());
                                CategoryScore {
                                    category,
                                    flagged,
                                    score: if flagged { 1.0 } else { 0.0 },
                                    applied: vec![Modality::Text],
                                }
                            })
                            .collect(),
                    )
                })
                .collect();
            Ok(Moderation {
                model: Self::MODEL_ID.to_owned(),
                results,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_echo_flags_the_categories_the_text_names() {
        let request = ModerationRequest {
            model: None,
            inputs: vec![
                vec![ModerationPart::Text("a calm sentence".into())],
                vec![ModerationPart::Text("this is flag:violence/graphic".into())],
            ],
        };
        let m = EchoModerationModel.moderate(request).await.unwrap();
        assert!(!m.results[0].flagged);
        assert!(m.results[1].flagged);
        let graphic = m.results[1]
            .get(ModerationCategory::ViolenceGraphic)
            .unwrap();
        assert_eq!((graphic.flagged, graphic.score), (true, 1.0));
        assert!(
            !m.results[1]
                .get(ModerationCategory::Violence)
                .unwrap()
                .flagged
        );
    }
}
