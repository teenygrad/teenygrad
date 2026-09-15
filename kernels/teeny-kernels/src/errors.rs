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

pub type Result<T> = anyhow::Result<T>;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// [`TileGraph::propagate`](crate::graph::optimizer::anduin::tile_graph::TileGraph::propagate)
    /// hit a node whose `tile_spec` declares zero outputs
    /// (`KernelTileSpec::outputs` is empty). Every tile-spec'd op needs at
    /// least one declared output to resolve its own required output tile
    /// against -- an empty `outputs` is always a spec-authoring mistake,
    /// not a legitimate op shape.
    #[error(
        "node {node} (\"{op_name}\")'s tile_spec declares no outputs -- \
         KernelTileSpec::outputs must have at least one entry"
    )]
    TileSpecMissingOutput { node: usize, op_name: String },

    /// A tile_spec's declared output rank disagrees with the real rank
    /// [`TileGraph::propagate`](crate::graph::optimizer::anduin::tile_graph::TileGraph::propagate)
    /// resolved for that output tile -- the spec was hand-authored
    /// against a different tensor rank than the op actually produces at
    /// runtime.
    #[error(
        "node {node} (\"{op_name}\")'s tile_spec declares output {output_index} at rank \
         {expected}, but its resolved tile has rank {actual}"
    )]
    OutputRankMismatch {
        node: usize,
        op_name: String,
        output_index: usize,
        expected: usize,
        actual: usize,
    },

    /// A tile_spec's declared input rank disagrees with the real rank of
    /// the edge feeding that input in the tile graph -- the spec was
    /// hand-authored against a different tensor rank than the op actually
    /// consumes at runtime.
    #[error(
        "node {node} (\"{op_name}\")'s tile_spec declares input {input_index} at rank \
         {expected}, but its edge's real shape has rank {actual}"
    )]
    InputRankMismatch {
        node: usize,
        op_name: String,
        input_index: usize,
        expected: usize,
        actual: usize,
    },

    /// A `TileAxisBinding::dims` is empty, so there's no real tensor axis
    /// for this binding to resolve against -- always a spec-authoring
    /// mistake (see `TileAxisBinding::dims`'s doc comment).
    #[error(
        "node {node} (\"{op_name}\")'s tile_spec axis \"{extent_param}\" has an empty `dims` list"
    )]
    EmptyAxisDims {
        node: usize,
        op_name: String,
        extent_param: &'static str,
    },

    #[error("Codegen error: {0}")]
    CodegenError(String),
}
