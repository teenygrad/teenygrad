/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

use std::str::ParseBoolError;

use thiserror::Error;

use crate::graph::OpType;
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

    #[error("Unsupported op: {0:?}")]
    UnsupportedOp(OpType),

    #[error("Parse error: {0}")]
    ParseBoolError(ParseBoolError),
}
