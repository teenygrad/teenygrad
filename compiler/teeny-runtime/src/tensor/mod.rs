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

use std::sync::Arc;

use teeny_core::dtype;

pub mod ops;

#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum Tensor<N: dtype::Dtype> {
    #[cfg(feature = "cpu")]
    Cpu(teeny_cpu::tensor::CpuTensor<N>),

    #[cfg(feature = "cuda")]
    Cuda(teeny_cuda::tensor::CudaTensor<N>),
}

pub type TensorRef<T> = Arc<Tensor<T>>;
