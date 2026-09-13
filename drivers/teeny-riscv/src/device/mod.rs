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
use std::sync::Arc;

use teeny_core::device::program::Kernel;
use teeny_core::device::{Device, LaunchConfig};
use teeny_core::dtype::Num;

use crate::device::buffer::{BufferRegistry, RiscvBuffer};
use crate::device::context::RiscvDeviceInfo;
use crate::device::program::RiscvProgram;
use crate::errors::Result;

/// Device-side memory buffers (host memory -- RISC-V kernels run against host-owned memory,
/// there is no separate device address space to copy across).
pub mod buffer;
/// Device/context management.
pub mod context;
/// Compiled kernel programs.
pub mod program;

/// A RISC-V kernel launch's configuration: the Triton launch grid.
///
/// [`RiscvDevice`]'s `launch` calls the kernel once for every program id in `grid`, the same
/// program ids a Triton launcher would pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RiscvLaunchConfig {
    /// The number of program ids along the x, y and z axes.
    pub grid: [u32; 3],
}

impl RiscvLaunchConfig {
    /// A launch over `grid` program ids.
    pub fn new(grid: [u32; 3]) -> Self {
        Self { grid }
    }
}

impl Default for RiscvLaunchConfig {
    /// A single program id.
    fn default() -> Self {
        Self::new([1, 1, 1])
    }
}

impl LaunchConfig for RiscvLaunchConfig {}

/// A RISC-V "device": there is no discrete accelerator to open, so this just represents the
/// local machine or the `qemu-riscv64` user-mode emulation environment kernels run under.
pub struct RiscvDevice<'a> {
    /// This device's (synthetic) static properties.
    pub info: RiscvDeviceInfo,
    /// The buffers this device has allocated, so `launch` can resolve pointer arguments.
    buffers: Arc<BufferRegistry>,
    _unused: PhantomData<&'a ()>,
}

impl<'a> RiscvDevice<'a> {
    /// Opens the (sole, synthetic) RISC-V device.
    pub fn new(info: RiscvDeviceInfo) -> Self {
        Self {
            info,
            buffers: Arc::default(),
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
        Ok(RiscvBuffer::with_registry(count, Arc::clone(&self.buffers)))
    }

    /// Runs `program` once for every program id in `cfg.grid`, under `qemu-riscv64`.
    ///
    /// Every pointer in `args` must point into a buffer allocated by this device; whatever the
    /// kernel writes to those buffers is copied back before this returns.
    #[cfg(feature = "qemu")]
    fn launch<K: Kernel>(
        &self,
        program: &Self::Program<K>,
        cfg: &Self::LaunchConfig,
        args: K::Args<'a>,
    ) -> Result<()> {
        crate::qemu::launch(
            &self.buffers,
            program.path(),
            program.entry_point(),
            cfg.grid,
            &args,
        )
    }

    /// Always panics: there is no RISC-V hardware support, and without the `qemu` feature there
    /// is nothing else to run the kernel on.
    #[cfg(not(feature = "qemu"))]
    fn launch<K: Kernel>(
        &self,
        _program: &Self::Program<K>,
        _cfg: &Self::LaunchConfig,
        _args: K::Args<'a>,
    ) -> Result<()> {
        panic!(
            "no RISC-V hardware to launch kernels on; enable teeny-riscv's `qemu` feature to run \
             them under qemu-riscv64"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::buffer::Region;
    use crate::device::context::Riscv;
    use teeny_core::device::buffer::Buffer;
    use teeny_core::device::context::Context;

    #[cfg(not(feature = "qemu"))]
    struct TestKernel;

    #[cfg(not(feature = "qemu"))]
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
    fn device_buffers_are_registered_until_dropped() {
        let device = RiscvDevice::new(RiscvDeviceInfo::default());
        let buf = device.buffer::<f32>(4).unwrap();
        let addr = buf.as_device_ptr() as usize;
        let region = Region { addr, bytes: 16 };

        assert_eq!(device.buffers.find(addr), Some(region));
        assert_eq!(device.buffers.find(addr + 12), Some(region));
        assert_eq!(device.buffers.find(addr + 16), None);

        drop(buf);
        assert_eq!(device.buffers.find(addr), None);
    }

    #[test]
    fn context_lists_exactly_one_synthetic_device() {
        let ctx = Riscv::try_new().unwrap();
        let devices = ctx.list_devices().unwrap();
        assert_eq!(devices.len(), 1);
    }

    #[test]
    #[cfg(not(feature = "qemu"))]
    #[should_panic(expected = "no RISC-V hardware")]
    fn launch_without_qemu_panics() {
        let program =
            RiscvProgram::<TestKernel>::from_parts("kernel.so", "test_kernel_entry_point");
        let device = RiscvDevice::new(RiscvDeviceInfo::default());
        let _ = device.launch(&program, &RiscvLaunchConfig::default(), ());
    }
}
