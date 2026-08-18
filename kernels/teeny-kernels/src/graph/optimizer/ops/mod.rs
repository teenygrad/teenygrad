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

//! Custom ops emitted by graph optimizers (composed kernels, not first-class `Op`s).

mod pointwise_fuse;
mod reduce_fuse;
mod shared_transpose_fuse;
mod tile_fuse;

pub use pointwise_fuse::{
    PointwiseFuse, is_bool_terminal_only, is_pointwise_fuse_dtype, probe_pointwise_op,
};
pub(crate) use reduce_fuse::choose_reduce_fuse_block_inner;
pub use reduce_fuse::{ReduceFuse, is_reduce_fuse_member, is_reduce_fuse_reducible};
pub use shared_transpose_fuse::SharedTransposeFuse;
pub(crate) use shared_transpose_fuse::{choose_shared_transpose_fuse_block_size, elem_bytes};
pub use tile_fuse::{TileFuse, is_tile_fuse_tail};
