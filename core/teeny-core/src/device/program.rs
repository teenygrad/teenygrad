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

use alloc::{format, string::String};
use sha2::{Digest, Sha256};

/// Receives each argument of a kernel in order. Implement this on a backend-
/// specific "packer" struct (e.g. a CUDA arg-pointer array builder).
pub trait ArgVisitor {
    fn visit_ptr(&mut self, ptr: *mut core::ffi::c_void);
    fn visit_i32(&mut self, val: i32);
    fn visit_u32(&mut self, val: u32);
    fn visit_f32(&mut self, val: f32);
}

/// Implemented by each concrete argument type; dispatches to the right
/// `ArgVisitor` method.
pub trait KernelArg {
    fn visit<V: ArgVisitor>(&self, visitor: &mut V);
}

impl<T> KernelArg for *mut T {
    #[inline]
    fn visit<V: ArgVisitor>(&self, visitor: &mut V) {
        visitor.visit_ptr(*self as *mut core::ffi::c_void);
    }
}

impl KernelArg for i32 {
    #[inline]
    fn visit<V: ArgVisitor>(&self, visitor: &mut V) {
        visitor.visit_i32(*self);
    }
}

impl KernelArg for u32 {
    #[inline]
    fn visit<V: ArgVisitor>(&self, visitor: &mut V) {
        visitor.visit_u32(*self);
    }
}

impl KernelArg for f32 {
    #[inline]
    fn visit<V: ArgVisitor>(&self, visitor: &mut V) {
        visitor.visit_f32(*self);
    }
}

/// Implemented for tuples of `KernelArg`s; visits each element in order.
/// The proc macro generates `type Args<'a> = (A, B, C, ...)` and the blanket
/// tuple impls below make that satisfy this bound automatically.
pub trait KernelArgs {
    fn visit_args<V: ArgVisitor>(&self, visitor: &mut V);
}

impl KernelArgs for () {
    #[inline]
    fn visit_args<V: ArgVisitor>(&self, _visitor: &mut V) {}
}

macro_rules! impl_kernel_args {
    ($( $T:ident : $idx:tt ),+) => {
        impl<$($T: KernelArg),+> KernelArgs for ($($T,)+) {
            #[inline]
            fn visit_args<V: ArgVisitor>(&self, visitor: &mut V) {
                $( self.$idx.visit(visitor); )+
            }
        }
    };
}

impl_kernel_args!(A:0);
impl_kernel_args!(A:0, B:1);
impl_kernel_args!(A:0, B:1, C:2);
impl_kernel_args!(A:0, B:1, C:2, D:3);
impl_kernel_args!(A:0, B:1, C:2, D:3, E:4);
impl_kernel_args!(A:0, B:1, C:2, D:3, E:4, F:5);
impl_kernel_args!(A:0, B:1, C:2, D:3, E:4, F:5, G:6);
impl_kernel_args!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7);
impl_kernel_args!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8);
impl_kernel_args!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9);
impl_kernel_args!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10);
impl_kernel_args!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11);
impl_kernel_args!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12);
impl_kernel_args!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12, N:13);
impl_kernel_args!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12, N:13, O:14);
impl_kernel_args!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12, N:13, O:14, P:15);
impl_kernel_args!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12, N:13, O:14, P:15, Q:16);
impl_kernel_args!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12, N:13, O:14, P:15, Q:16, R:17);
impl_kernel_args!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12, N:13, O:14, P:15, Q:16, R:17, S:18);
impl_kernel_args!(A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11, M:12, N:13, O:14, P:15, Q:16, R:17, S:18, Ty:19);

pub trait Kernel {
    type Args<'a>: KernelArgs;

    fn id(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.source().as_bytes());
        hasher.finalize().iter().map(|b| format!("{:02x}", b)).collect()
    }

    fn name(&self) -> &str;

    fn source(&self) -> &str;

    fn kernel_source(&self) -> &str;

    fn entry_point(&self) -> &str;
}

pub trait Program<'a, K: Kernel>: Sized {}
