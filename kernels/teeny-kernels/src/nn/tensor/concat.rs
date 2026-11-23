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

//! `Concat` along any axis of any rank.
//!
//! There is no kernel here, and there does not need to be one: concatenation along an
//! axis is a contiguous copy per leading block, whatever the rank and whichever axis it
//! is. Writing `outer` for the product of the axes in front of the concatenated one and
//! `inner` for the product of those behind it, every input's block for a fixed leading
//! coordinate is `extent * inner` contiguous elements, and it lands in a contiguous run
//! of the output of the same length. That is exactly the mapping
//! [`super::channel_cat::channel_cat_forward`] performs, with `chunk_c` standing for
//! `extent * inner` and `c_total` for the output's whole axis times `inner`, so this
//! reuses that kernel and only works out its arguments.
//!
//! One launch per input, each writing a disjoint run of the output, which is what the
//! `RuntimeOp` multi-launch path is for.

use teeny_core::dtype::Num;
use teeny_core::model::{RawPtr, RuntimeOp};

use super::channel_cat::ChannelCatForward;

/// How an input's block maps into the output, in elements.
struct Block {
    /// Leading blocks: the product of the axes in front of the concatenated one.
    outer: usize,
    /// This input's contribution per leading block.
    chunk: usize,
    /// The output's whole run per leading block.
    total: usize,
    /// Where this input's contribution starts inside that run.
    offset: usize,
}

/// Dispatch for `Concat`: one launch per input, each a contiguous copy.
pub struct ConcatRuntimeOp<D: Num + Send + Sync + 'static> {
    kernel: ChannelCatForward<D>,
    /// The axis being concatenated, already normalised against the rank.
    axis: usize,
    n_inputs: usize,
}

impl<D: Num + Send + Sync + 'static> ConcatRuntimeOp<D> {
    /// `axis` is a non-negative axis index, and `n_inputs` the number of operands.
    pub fn new(block_size: i32, axis: usize, n_inputs: usize) -> Option<Self> {
        if n_inputs == 0 {
            return None;
        }
        Some(Self {
            kernel: ChannelCatForward::<D>::new(block_size),
            axis,
            n_inputs,
        })
    }

    /// The generated kernel, for its name and source.
    pub fn kernel(&self) -> &ChannelCatForward<D> {
        &self.kernel
    }

    /// Where input `which` of `shapes` goes in the output.
    ///
    /// The shapes are the ones the executor hands over at launch time, which are the
    /// same ones lowering checked, so a disagreement between them cannot arise here.
    fn block(&self, which: usize, shapes: &[&[usize]]) -> Block {
        let shape = shapes[which];
        let outer: usize = shape[..self.axis].iter().product();
        let inner: usize = shape[self.axis + 1..].iter().product();
        let chunk = shape[self.axis] * inner;
        let total: usize = shapes.iter().map(|s| s[self.axis] * inner).sum();
        let offset: usize = shapes[..which].iter().map(|s| s[self.axis] * inner).sum();
        Block {
            outer,
            chunk,
            total,
            offset,
        }
    }

    fn launch_grid(&self, which: usize, shapes: &[&[usize]]) -> [u32; 3] {
        let b = self.block(which, shapes);
        let tiles = b.chunk.div_ceil(self.kernel.block_size as usize);
        [(b.outer * tiles) as u32, 1, 1]
    }
}

impl<D: Num + Send + Sync + 'static> RuntimeOp for ConcatRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        self.n_inputs
    }

    fn param_shapes(&self, _: &[&[usize]], _: &[usize]) -> Vec<Vec<usize>> {
        Vec::new()
    }

    fn n_launches(&self) -> usize {
        self.n_inputs
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        // Reached only for a single-input concat, which is a copy of that input.
        self.pack_args_for_launch(
            0,
            inputs,
            params,
            output,
            output_shape,
            output_row_stride,
            visitor,
        );
    }

    fn pack_args_for_launch(
        &self,
        launch_idx: usize,
        inputs: &[(RawPtr, &[usize])],
        _params: &[RawPtr],
        output: RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn teeny_core::device::program::ArgVisitor,
    ) {
        let shapes: Vec<&[usize]> = inputs.iter().map(|(_, s)| *s).collect();
        let b = self.block(launch_idx, &shapes);
        visitor.visit_ptr(inputs[launch_idx].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(b.chunk as i32);
        visitor.visit_i32(b.total as i32);
        visitor.visit_i32(b.offset as i32);
    }

    fn grid_for_launch(
        &self,
        launch_idx: usize,
        input_shapes: &[&[usize]],
        _output_shape: &[usize],
    ) -> [u32; 3] {
        self.launch_grid(launch_idx, input_shapes)
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // Reached only for a single-input concat, where the executor takes the
        // single-launch path: the output's shape is that input's shape, so the grid
        // is the same one `grid_for_launch` would work out.
        self.launch_grid(0, &[output_shape])
    }
}
