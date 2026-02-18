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

use std::fmt::Display;

#[cfg(feature = "cpu")]
use teeny_cpu::target::CpuTarget;

#[cfg(feature = "cuda")]
use teeny_cuda::target::CudaTarget;

#[derive(Debug, Clone)]
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

impl Display for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Target::Cpu(target) => write!(f, "{}", target),
            Target::Cuda(target) => write!(f, "nvidia-{}", target),
        }
    }
}
