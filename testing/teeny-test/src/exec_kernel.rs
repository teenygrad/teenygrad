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

use teeny_core::device::program::Kernel;
use teeny_core::model::ExecutableOp;

/// Adapts a lowered [`ExecutableOp`] to [`Kernel`], for any backend's `compile_kernel`
/// (e.g. `teeny_cuda::compiler::compile_kernel`, `teeny_riscv::compiler::compile_kernel`).
pub struct ExecKernel<'a>(pub &'a dyn ExecutableOp);

impl Kernel for ExecKernel<'_> {
    type Args<'b> = ();

    fn name(&self) -> &str {
        self.0.name()
    }

    fn source(&self) -> &str {
        self.0.forward_kernel_source()
    }

    fn kernel_source(&self) -> &str {
        self.0.forward_kernel_source()
    }

    fn entry_point_source(&self) -> &str {
        ""
    }

    fn entry_point_name(&self) -> String {
        self.0.forward_kernel_entry_point().to_string()
    }
}
