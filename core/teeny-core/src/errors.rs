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

    /// A [`crate::model::tile_spec::KernelTileSpec`] contradicts itself.
    ///
    /// These are spec-authoring bugs. The consumer's documented behaviour
    /// for each is to skip or overwrite silently, so nothing surfaces them
    /// at the point they are written — hence the explicit check.
    #[error("tile spec for `{param}`: {problem}")]
    InvalidTileSpec {
        /// The tensor parameter the bad spec describes, e.g. `"x_ptr"`.
        param: String,
        /// What is wrong, naming the offending index or name.
        problem: String,
    },
}
