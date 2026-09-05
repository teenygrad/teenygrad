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

use std::marker::PhantomData;

use teeny_core::device::program::Kernel;
use teeny_core::device::{Device, LaunchConfig};
use teeny_core::dtype::Num;

use crate::device::buffer::RiscvBuffer;
use crate::device::context::RiscvDeviceInfo;
use crate::device::program::RiscvProgram;
use crate::errors::{Error, Result};

/// Device-side memory buffers (host `Vec`s -- RISC-V kernels run against host-owned memory,
/// there is no separate device address space to copy across).
pub mod buffer;
/// Device/context management.
pub mod context;
/// Compiled kernel programs.
pub mod program;

/// A RISC-V kernel launch's configuration.
///
/// Empty for now: no real grid/block scheduling exists yet (the compiler backend always emits a
/// single no-argument placeholder function -- see [`crate::errors::Error::ArgumentPassingNotSupported`]).
#[derive(Debug, Default, Clone, Copy)]
pub struct RiscvLaunchConfig;

impl LaunchConfig for RiscvLaunchConfig {}

/// A RISC-V "device": there is no discrete accelerator to open, so this just represents the
/// local machine (native RISC-V) or the `qemu-riscv64` user-mode emulation environment this
/// process is running under/targeting.
pub struct RiscvDevice<'a> {
    /// This device's (synthetic) static properties.
    pub info: RiscvDeviceInfo,
    _unused: PhantomData<&'a ()>,
}

impl<'a> RiscvDevice<'a> {
    /// Opens the (sole, synthetic) RISC-V device.
    pub fn new(info: RiscvDeviceInfo) -> Self {
        Self {
            info,
            _unused: PhantomData,
        }
    }

    /// This device's static properties.
    pub fn info(&self) -> &RiscvDeviceInfo {
        &self.info
    }
}

impl<'a> Device<'a> for RiscvDevice<'a> {
    type Buffer<N: Num> = RiscvBuffer<'a, N>;
    type Program<K: Kernel> = RiscvProgram<'a, K>;
    type LaunchConfig = RiscvLaunchConfig;

    /// Allocates a zero-initialized host buffer -- host memory *is* the device memory here.
    fn buffer<N: Num>(&self, count: usize) -> Result<Self::Buffer<N>> {
        RiscvBuffer::try_new(count)
    }

    /// Not supported yet: see [`Error::ArgumentPassingNotSupported`]'s doc comment. The
    /// compiler backend (`RiscvBackend`) always emits the same no-argument placeholder function
    /// regardless of `program`'s actual kernel body, so there is no real per-kernel argument ABI
    /// to marshal `args` into. Load and call the placeholder directly via
    /// [`crate::runtime::KernelLibrary::call_void_kernel`] instead, as
    /// `tests/test_qemu_relu.rs` does, until `teenygrad-1zd`'s compiler-side work lands.
    fn launch<K: Kernel>(
        &self,
        _program: &Self::Program<K>,
        _cfg: &Self::LaunchConfig,
        _args: K::Args<'a>,
    ) -> Result<()> {
        Err(Error::ArgumentPassingNotSupported.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::context::Riscv;
    use teeny_core::device::buffer::Buffer;
    use teeny_core::device::context::Context;

    struct TestKernel;

    impl Kernel for TestKernel {
        type Args<'a> = ();

        fn name(&self) -> &str {
            "test_kernel"
        }

        fn source(&self) -> &str {
            ""
        }

        fn kernel_source(&self) -> &str {
            ""
        }

        fn entry_point_source(&self) -> &str {
            ""
        }
    }

    #[test]
    fn buffer_round_trips_to_device_and_to_host() {
        let device = RiscvDevice::new(RiscvDeviceInfo::default());
        let mut buf = device.buffer::<f32>(4).unwrap();

        let input = [1.0f32, 2.0, 3.0, 4.0];
        buf.to_device(&input).unwrap();

        let mut output = [0.0f32; 4];
        buf.to_host(&mut output).unwrap();

        assert_eq!(input, output);
    }

    #[test]
    fn context_lists_exactly_one_synthetic_device() {
        let ctx = Riscv::try_new().unwrap();
        let devices = ctx.list_devices().unwrap();
        assert_eq!(devices.len(), 1);
    }

    #[test]
    fn launch_is_not_yet_supported() {
        // `libc.so.6` stands in for a compiled kernel .so here -- `launch`'s stub doesn't
        // inspect the loaded library at all, so any loadable library exercises the same path a
        // real (RISC-V) kernel .so would.
        let program = RiscvProgram::<TestKernel>::try_new("libc.so.6").unwrap();
        let device = RiscvDevice::new(RiscvDeviceInfo::default());

        let err = device.launch(&program, &RiscvLaunchConfig, ()).unwrap_err();
        assert!(matches!(
            err.downcast_ref::<Error>(),
            Some(Error::ArgumentPassingNotSupported)
        ));
    }
}
