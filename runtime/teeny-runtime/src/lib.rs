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

//! Feature-selected [`teeny_core::device`] backend for [teenygrad](https://teenygrad.org):
//! [`Context`]/[`Device`]/[`LaunchConfig`] resolve to `teeny-cuda`'s or `teeny-riscv`'s concrete
//! types depending on which of this crate's `cuda`/`riscv` features is enabled, so calling code
//! can be written once against `teeny_runtime::{Context, Device, ...}` instead of hardcoding one
//! backend. Enable exactly one of `cuda`/`riscv` -- enabling both is a compile error (`backend`
//! defined twice), and enabling neither fails with the `compile_error!` below.
//!
//! Not to be confused with `teeny_riscv::runtime` (a `dlopen` wrapper module inside the RISC-V
//! driver crate) -- this is a different, top-level crate.
//!
//! `teeny-riscv`'s `Device::launch` is a structured-error stub today (the compiler backend has
//! no real per-kernel argument ABI yet -- see `teenygrad-1zd`), so generic code written against
//! this crate should expect `launch` to fail on the `riscv` backend until that lands.

#[cfg(not(any(feature = "cuda", feature = "riscv")))]
compile_error!("teeny-runtime needs exactly one of the `cuda` or `riscv` features enabled");

#[cfg(feature = "cuda")]
mod backend {
    /// The compiled-in backend's [`teeny_core::device::context::Context`] type.
    pub type Context<'a> = teeny_cuda::device::context::Cuda<'a>;
    /// The compiled-in backend's [`teeny_core::device::Device`] type.
    pub type Device<'a> = teeny_cuda::device::CudaDevice<'a>;
    /// The compiled-in backend's [`teeny_core::device::LaunchConfig`] type.
    pub type LaunchConfig = teeny_cuda::device::CudaLaunchConfig;

    pub(crate) fn context<'a>() -> anyhow::Result<Context<'a>> {
        Context::try_new()
    }
}

#[cfg(feature = "riscv")]
mod backend {
    /// The compiled-in backend's [`teeny_core::device::context::Context`] type.
    pub type Context<'a> = teeny_riscv::device::context::Riscv<'a>;
    /// The compiled-in backend's [`teeny_core::device::Device`] type.
    pub type Device<'a> = teeny_riscv::device::RiscvDevice<'a>;
    /// The compiled-in backend's [`teeny_core::device::LaunchConfig`] type.
    pub type LaunchConfig = teeny_riscv::device::RiscvLaunchConfig;

    pub(crate) fn context<'a>() -> anyhow::Result<Context<'a>> {
        Context::try_new()
    }
}

pub use backend::{Context, Device, LaunchConfig};

/// Opens the compiled-in backend's [`Context`], and opens its first listed device.
pub fn open<'a>() -> anyhow::Result<Device<'a>> {
    use teeny_core::device::context::{Context as _, DeviceInfo as _};

    let ctx = backend::context()?;
    let devices = ctx.list_devices()?;
    let first = devices
        .first()
        .ok_or_else(|| anyhow::anyhow!("no devices found"))?;
    ctx.device(&first.id())
}
