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

use std::str::ParseBoolError;

use thiserror::Error;

use crate::torch::DType;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Flatbuffers error: {0}")]
    InvalidFlatbuffer(#[from] flatbuffers::InvalidFlatbuffer),

    #[error("Invalid buffer: {0}")]
    InvalidBuffer(String),

    #[error("Deserialization failed: {0}")]
    DeserializationFailed(String),

    #[error("No graph nodes")]
    NoGraphNodes,

    #[error("No graph node: {0}")]
    NoGraphNode(String),

    #[error("No graph node name: {0}")]
    NoGraphNodeName(String),

    #[error("No graph node target: {0}")]
    NoGraphNodeTarget(String),

    #[error("No graph node args: {0}")]
    NoGraphNodeArgs(String),

    #[error("Graph node missing args: {0}")]
    GraphNodeMissingArgs(String),

    #[error("Graph node invalid args: {0}")]
    GraphNodeInvalidArgs(String),

    #[error("Graph node not found: {0}")]
    GraphNodeNotFound(String),

    #[error("No matching function: {0}")]
    NoMatchingFunction(String),

    #[error("Parse error: {0}")]
    ParseBoolError(ParseBoolError),

    #[error("Unsupported dtype: {0:?}")]
    UnsupportedDtype(DType),

    #[error("Parsing device error: {0}")]
    ParsingDevice(String),

    #[error("FXGraph error: {0}")]
    FxGraph(teeny_fxgraph::errors::Error),
}
