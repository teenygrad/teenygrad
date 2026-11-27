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

use serde::{Deserialize, Serialize};

/// `Model`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    /// Always `"model"`.
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

impl From<crate::ModelInfo> for Model {
    fn from(m: crate::ModelInfo) -> Self {
        Self {
            id: m.id,
            object: "model".to_owned(),
            created: m.created,
            owned_by: m.owned_by,
        }
    }
}

/// `ListModelsResponse`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListModelsResponse {
    /// Always `"list"`.
    pub object: String,
    pub data: Vec<Model>,
}
