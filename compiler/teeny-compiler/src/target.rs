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

#[cfg(feature = "cpu")]
use teeny_cpu::target::Target as CpuTarget;

#[cfg(feature = "cuda")]
use teeny_cuda::target::Target as CudaTarget;

pub enum Target {
    #[cfg(feature = "cpu")]
    Cpu(CpuTarget),

    #[cfg(feature = "cuda")]
    Cuda(CudaTarget),
}

impl From<CpuTarget> for Target {
    fn from(target: CpuTarget) -> Self {
        Target::Cpu(target)
    }
}

impl From<CudaTarget> for Target {
    fn from(target: CudaTarget) -> Self {
        Target::Cuda(target)
    }
}
