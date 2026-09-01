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

//! Test-only helpers shared across this module's other files' `tests`
//! submodules: a minimal [`ExecutableOp`] test double, and a couple of
//! [`KernelTileSpec`]s exercised by more than one of them.

#![cfg(test)]

use teeny_core::graph::{DtypeRepr, Shape};
use teeny_core::model::{ExecutableOp, KernelTileSpec, TensorTileSpec, TileAxisBinding};

use super::{EdgeId, NodeId, TileGraph};

/// Minimal [`ExecutableOp`] test double: just enough surface
/// (name/shape/dtype/tile_spec) for [`super::TileGraph::from_dag`] to convert on.
struct TestOp {
    name: &'static str,
    dtype: DtypeRepr,
    shape: Shape,
    is_input: bool,
    tile_spec: Option<KernelTileSpec>,
}

impl ExecutableOp for TestOp {
    fn name(&self) -> &str {
        self.name
    }

    fn is_input(&self) -> bool {
        self.is_input
    }

    fn forward_kernel_source(&self) -> &str {
        ""
    }

    fn forward_kernel_entry_point(&self) -> &str {
        ""
    }

    fn output_shape(&self) -> &Shape {
        &self.shape
    }

    fn output_dtype(&self) -> DtypeRepr {
        self.dtype
    }

    fn tile_spec(&self) -> Option<KernelTileSpec> {
        self.tile_spec
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub(super) fn op(name: &'static str, shape: Shape, is_input: bool) -> Box<dyn ExecutableOp> {
    Box::new(TestOp {
        name,
        dtype: DtypeRepr::F32,
        shape,
        is_input,
        tile_spec: None,
    })
}

pub(super) fn op_with_tile_spec(
    name: &'static str,
    shape: Shape,
    is_input: bool,
    tile_spec: KernelTileSpec,
) -> Box<dyn ExecutableOp> {
    Box::new(TestOp {
        name,
        dtype: DtypeRepr::F32,
        shape,
        is_input,
        tile_spec: Some(tile_spec),
    })
}

/// A flat, single-axis elementwise spec: input and output share one
/// `extent_param` name (`"n_elements"`), so resolving the output
/// resolves the input with no arithmetic at all.
pub(super) fn flat_unary_spec() -> KernelTileSpec {
    const AXIS: TileAxisBinding = TileAxisBinding {
        dims: &[0],
        block_const: "BLOCK_SIZE",
        extent_param: "n_elements",
        window: None,
        divide_by: None,
    };
    const X: TensorTileSpec = TensorTileSpec {
        param: "x_ptr",
        rank: 1,
        axes: &[AXIS],
        reduction_axis: None,
        untiled_dims: &[],
    };
    const Y: TensorTileSpec = TensorTileSpec {
        param: "y_ptr",
        ..X
    };
    KernelTileSpec {
        inputs: &[X],
        outputs: &[Y],
        loop_spec: None,
    }
}

/// The [`EdgeId`] of the edge from `producer` to `consumer` in
/// `tile_graph`, panicking if `producer` isn't one of `consumer`'s parents.
/// Shared by tests that need to look up one specific internal edge to
/// assert against, instead of repeating the `parent_edges`
/// find/map/expect dance at each call site.
pub(super) fn edge_between(tile_graph: &TileGraph, producer: NodeId, consumer: NodeId) -> EdgeId {
    tile_graph
        .parent_edges(consumer)
        .into_iter()
        .find(|&(p, _)| p == producer)
        .map(|(_, id)| id)
        .unwrap_or_else(|| panic!("no edge from node {producer} to node {consumer}"))
}

/// A batchnorm2d-style spec: mirrors the real
/// `batch_norm_2d_nchw_forward_inference` kernel (grid `[C, B]`, one
/// `BLOCK_HW`-wide loop over the *flattened* `H*W` range per CTA) --
/// the case `TileAxisBinding::dims` having more than one entry exists
/// for (teenygrad-1nr.8). NCHW: dim 0 = B, dim 1 = C (both untiled,
/// grid-driven), dims 2/3 = H/W flattened into one binding (`dims:
/// &[2, 3]`, W innermost, matching NCHW's row-major layout). Input
/// and output share `"HW"`, so this is shape-preserving elementwise
/// like `flat_unary_spec`, just spanning a flattened pair of real
/// dims instead of one.
pub(super) fn batchnorm2d_shaped_spec() -> KernelTileSpec {
    const HW: TileAxisBinding = TileAxisBinding {
        dims: &[2, 3],
        block_const: "BLOCK_HW",
        extent_param: "HW",
        window: None,
        divide_by: None,
    };
    const X: TensorTileSpec = TensorTileSpec {
        param: "x_ptr",
        rank: 4,
        axes: &[HW],
        reduction_axis: None,
        untiled_dims: &["B", "C"],
    };
    const Y: TensorTileSpec = TensorTileSpec {
        param: "y_ptr",
        ..X
    };
    KernelTileSpec {
        inputs: &[X],
        outputs: &[Y],
        loop_spec: None,
    }
}
