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

use alloc::string::String;

/// `teeny-core`'s result alias.
pub type Result<T> = anyhow::Result<T>;

/// Errors produced by `teeny-core`.
#[derive(thiserror::Error, Debug)]
pub enum Error {
    /// A context/name-scope error.
    #[error("Context error: {0}")]
    ContextError(String),

    /// A sliding-window op (convolution or pooling) was given a zero stride,
    /// which would divide by zero while placing windows.
    #[error(
        "{op}: {axis} stride is 0, but a sliding window has to advance by at \
         least one element per step. Set the {axis} stride to 1 or more."
    )]
    ZeroWindowStride {
        /// The op that owns the window, e.g. `"Conv2d"`.
        op: String,
        /// The axis the stride applies to, e.g. `"height"`.
        axis: String,
    },

    /// A sliding-window op's kernel is wider than its padded input, so no
    /// window position is valid along that axis.
    #[error(
        "{op}: {axis} kernel {kernel} does not fit its input. The {axis} extent \
         is {extent}, and padding {padding} widens it to only {}, so no \
         window position is valid. Reduce the {axis} kernel to at most {}, \
         or raise the {axis} padding to at least {}.",
        .extent + 2 * .padding,
        .extent + 2 * .padding,
        (.kernel - .extent).div_ceil(2)
    )]
    WindowDoesNotFit {
        /// The op that owns the window, e.g. `"Conv2d"`.
        op: String,
        /// The axis the window runs along, e.g. `"height"`.
        axis: String,
        /// The input extent along `axis`, before padding.
        extent: usize,
        /// The kernel size along `axis`.
        kernel: usize,
        /// The padding applied to each side of `axis`.
        padding: usize,
    },

    /// A sliding-window op was asked to recover an input extent from an output
    /// extent its forward pass could never produce.
    #[error(
        "{op}: {axis} output extent {out} is not reachable. With kernel \
         {kernel}, stride {stride} and padding {padding}, the smallest output \
         this op can produce is {}, so no input extent maps to {out}. Check \
         the {axis} extent of the shape being reversed.",
        // `out = floor((in + 2p - k) / s) + 1` is smallest at `in = 0`; the
        // saturating subtraction covers a kernel wider than the padding
        // alone, and `max(1)` keeps a hand-built zero stride out of the
        // division.
        (2 * *.padding).saturating_sub(*.kernel) / (*.stride).max(1) + 1
    )]
    WindowOutputUnreachable {
        /// The op that owns the window, e.g. `"Conv2d"`.
        op: String,
        /// The axis the window runs along, e.g. `"height"`.
        axis: String,
        /// The output extent along `axis` that could not be reversed.
        out: usize,
        /// The kernel size along `axis`.
        kernel: usize,
        /// The stride along `axis`.
        stride: usize,
        /// The padding applied to each side of `axis`.
        padding: usize,
    },

    /// Reversing an op's shape would need an extent too large for `usize`.
    #[error(
        "{op}: reversing axis {axis} needs an extent of {extent} x {factor}, \
         which overflows usize. A shape that large cannot have been produced \
         by this op."
    )]
    ShapeExtentOverflow {
        /// The op being reversed, e.g. `"Split"`.
        op: String,
        /// The axis whose extent overflowed.
        axis: usize,
        /// The output extent along `axis`.
        extent: usize,
        /// The factor `extent` had to be multiplied by.
        factor: usize,
    },

    /// An op was handed a shape whose rank it cannot work with.
    #[error(
        "{op}: expected a rank-{expected} shape, got rank {actual}. {op} \
         operates on rank-{expected} tensors, so a rank-{actual} shape cannot \
         be one of its shapes."
    )]
    ShapeRankMismatch {
        /// The op that rejected the shape, e.g. `"Conv2d"`.
        op: String,
        /// The rank the op requires.
        expected: usize,
        /// The rank it was given.
        actual: usize,
    },
}
