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

//! Static I/O layout of a `#[kernel]` function (host-side only).
//!
//! Derived by `teeny_macros::kernel` from marked pointer params
//! ([`crate::triton::InPtr`] / [`crate::triton::OutPtr`] / …). Lives outside
//! `triton/` so it is not embedded into the no_core DSL string.

/// Role of one pointer parameter on a `#[kernel]` signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PtrRole {
    /// [`crate::triton::InPtr`] — read-only device pointer.
    In,
    /// [`crate::triton::OutPtr`] — write-only device pointer.
    Out,
    /// [`crate::triton::InOutPtr`] — read+write / in-place device pointer.
    InOut,
    /// Unmarked `T::Pointer<_>` (no In/Out wrapper).
    Raw,
}

/// Pointer-parameter layout of a kernel, in signature order (scalars omitted).
///
/// Fusion uses [`Self::roles`] to wire member *N*'s outputs to member *N+1*'s
/// inputs by argument index; counts are derived helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelIo {
    /// Role of each pointer parameter, in the order they appear in the signature.
    pub roles: &'static [PtrRole],
}

impl KernelIo {
    /// Number of [`PtrRole::In`] pointer parameters.
    pub const fn n_in(self) -> usize {
        self.count_role(PtrRole::In)
    }

    /// Number of [`PtrRole::Out`] pointer parameters.
    pub const fn n_out(self) -> usize {
        self.count_role(PtrRole::Out)
    }

    /// Number of [`PtrRole::InOut`] pointer parameters.
    pub const fn n_inout(self) -> usize {
        self.count_role(PtrRole::InOut)
    }

    /// Number of [`PtrRole::Raw`] pointer parameters.
    pub const fn n_unmarked(self) -> usize {
        self.count_role(PtrRole::Raw)
    }

    /// True when the kernel is a single-input / single-output pointer contract
    /// with no in-place or unmarked pointer args — the shape case-1 elementwise
    /// fusion selects on.
    pub const fn is_unary_elementwise(self) -> bool {
        matches!(self.roles, [PtrRole::In, PtrRole::Out])
    }

    /// Index of the first [`PtrRole::In`] pointer, if any.
    pub const fn first_in(self) -> Option<usize> {
        self.first_role(PtrRole::In)
    }

    /// Index of the first [`PtrRole::Out`] pointer, if any.
    pub const fn first_out(self) -> Option<usize> {
        self.first_role(PtrRole::Out)
    }

    const fn count_role(self, role: PtrRole) -> usize {
        let mut n = 0usize;
        let mut i = 0usize;
        while i < self.roles.len() {
            if role_eq(self.roles[i], role) {
                n += 1;
            }
            i += 1;
        }
        n
    }

    const fn first_role(self, role: PtrRole) -> Option<usize> {
        let mut i = 0usize;
        while i < self.roles.len() {
            if role_eq(self.roles[i], role) {
                return Some(i);
            }
            i += 1;
        }
        None
    }
}

const fn role_eq(a: PtrRole, b: PtrRole) -> bool {
    matches!(
        (a, b),
        (PtrRole::In, PtrRole::In)
            | (PtrRole::Out, PtrRole::Out)
            | (PtrRole::InOut, PtrRole::InOut)
            | (PtrRole::Raw, PtrRole::Raw)
    )
}

/// Pointer-layout metadata from a `#[kernel]` signature.
///
/// `#[kernel]` implements this for every generated kernel struct.
pub trait KernelIoLayout {
    /// Pointer-parameter layout in signature order.
    fn kernel_io() -> KernelIo;
}

/// Kernels that expose a `BLOCK_SIZE` (or equivalent) CTA tile size.
///
/// `#[kernel]` implements this when the annotated fn has `const BLOCK_SIZE: i32`.
pub trait BlockSized {
    /// CTA tile size used for the standard `ceil(n / BLOCK_SIZE)` grid.
    fn block_size(&self) -> i32;
}

/// Marker: host ABI ends with `n_elements: i32` and uses the standard tiled grid.
///
/// Softmax-style kernels that also have `BLOCK_SIZE` and unary `In`/`Out` do
/// **not** get this marker (their last scalar is `n_cols`, not `n_elements`).
pub trait NElementsTiled {}

/// Probe returned by [`PointwiseFuseProbeExt::pointwise_fuse_probe`].
///
/// Compatibility is CTA-identity: same tile size implies the same default grid
/// class for `n_elements`-tiled unary kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PointwiseFuseProbe {
    /// CTA tile size (`BLOCK_SIZE`).
    pub block_size: i32,
}

impl PointwiseFuseProbe {
    /// Whether two probes may share a fused CTA map.
    pub const fn compatible(self, other: Self) -> bool {
        self.block_size == other.block_size
    }

    /// Default grid for an `n_elements`-tiled pointwise kernel.
    pub fn grid(self, n_elements: usize) -> [u32; 3] {
        [n_elements.div_ceil(self.block_size as usize) as u32, 1, 1]
    }
}

/// Pointwise-fusion probe derived from kernel ABI metadata.
///
/// Blanket-implemented for every kernel that is [`KernelIoLayout`] +
/// [`BlockSized`] + [`NElementsTiled`]. Probe logic lives here — not in the
/// `#[kernel]` macro.
pub trait PointwiseFuseProbeExt {
    /// `Some` when this kernel is unary elementwise with the standard
    /// `n_elements` CTA map; otherwise `None`.
    fn pointwise_fuse_probe(&self) -> Option<PointwiseFuseProbe>;
}

impl<K: KernelIoLayout + BlockSized + NElementsTiled> PointwiseFuseProbeExt for K {
    fn pointwise_fuse_probe(&self) -> Option<PointwiseFuseProbe> {
        if K::kernel_io().is_unary_elementwise() {
            Some(PointwiseFuseProbe {
                block_size: self.block_size(),
            })
        } else {
            None
        }
    }
}
