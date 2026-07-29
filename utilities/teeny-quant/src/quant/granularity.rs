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

//! How many independent scale/zero-point pairs a quantized tensor gets.

/// Quantization granularity: how a tensor's elements are partitioned into groups that each get
/// their own scale/zero-point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Granularity {
    /// One scale/zero-point for the whole tensor.
    PerTensor,

    /// One scale/zero-point per index along `axis` (e.g. per output channel of a `Linear` or
    /// `Conv` weight). Equivalent to [`Granularity::Group`] with `group_size: 1`.
    PerChannel {
        /// Axis to quantize independently along (typically the output-channel axis, `0`).
        axis: usize,
    },

    /// One scale/zero-point per contiguous `group_size`-element block along `axis` (the
    /// GPTQ/AWQ/compressed-tensors convention; `axis` is typically the reduction/input-channel
    /// axis).
    Group {
        /// Axis the grouping runs along.
        axis: usize,
        /// Number of elements per group along `axis`.
        group_size: usize,
    },
}

impl Granularity {
    /// The `(axis, group_size)` this granularity reduces to, or `None` for [`Granularity::PerTensor`].
    pub(crate) fn axis_and_group_size(self) -> Option<(usize, usize)> {
        match self {
            Granularity::PerTensor => None,
            Granularity::PerChannel { axis } => Some((axis, 1)),
            Granularity::Group { axis, group_size } => Some((axis, group_size)),
        }
    }

    /// A short, stable name for the compressed-tensors `strategy` field.
    pub fn strategy_name(self) -> &'static str {
        match self {
            Granularity::PerTensor => "tensor",
            Granularity::PerChannel { .. } => "channel",
            Granularity::Group { .. } => "group",
        }
    }
}
