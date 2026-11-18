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

impl From<teeny_llm_core::ModelInfo> for Model {
    fn from(m: teeny_llm_core::ModelInfo) -> Self {
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
