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

//! Feature-selected [`teeny_core::device`]/[`teeny_core::compiler`] backend for
//! [teenygrad](https://teenygrad.org): [`Context`]/[`Device`]/[`LaunchConfig`]/[`Target`]/
//! [`Program`], plus [`compile_kernel`]/[`load_program`]/[`launch_config`]/[`open`], resolve to
//! `teeny-cuda`'s or `teeny-riscv`'s concrete types depending on which of this crate's
//! `cuda`/`riscv` features is enabled, so calling code can be written once against
//! `teeny_runtime::*` instead of hardcoding one backend.
//!
//! Enable at least one of `cuda`/`riscv` -- enabling neither fails with the `compile_error!`
//! below. Enabling both is **not** an error: `cuda` takes priority (this matches
//! `teeny-kernels`, whose `cuda` feature is on by default, so its `riscv` integration tests
//! don't have to also disable `cuda` to build).
//!
//! Not to be confused with `teeny_riscv::runtime` (a `dlopen` wrapper module inside the RISC-V
//! driver crate) -- this is a different, top-level crate.
//!
//! `teeny-riscv`'s `Device::launch` is a structured-error stub today (the compiler backend has
//! no real per-kernel argument ABI yet -- see `teenygrad-1zd`), so generic code written against
//! this crate should expect `launch` to fail on the `riscv` backend until that lands.

#[cfg(not(any(feature = "cuda", feature = "riscv")))]
compile_error!("teeny-runtime needs at least one of the `cuda` or `riscv` features enabled");

#[cfg(feature = "cuda")]
mod backend {
    use teeny_core::device::program::Kernel;

    /// The compiled-in backend's [`teeny_core::device::context::Context`] type.
    pub type Context<'a> = teeny_cuda::device::context::Cuda<'a>;
    /// The compiled-in backend's [`teeny_core::device::Device`] type.
    pub type Device<'a> = teeny_cuda::device::CudaDevice<'a>;
    /// The compiled-in backend's [`teeny_core::device::LaunchConfig`] type.
    pub type LaunchConfig = teeny_cuda::device::CudaLaunchConfig;
    /// The compiled-in backend's [`teeny_core::compiler::Target`] type.
    pub type Target = teeny_cuda::compiler::target::Target;
    /// The compiled-in backend's [`teeny_core::device::program::Program`] type.
    pub type Program<'a, K> = teeny_cuda::device::program::CudaProgram<'a, K>;

    /// Compiles `kernel` for `target` (see `teeny_cuda::compiler::compile_kernel`).
    pub use teeny_cuda::compiler::compile_kernel;

    pub(crate) fn context<'a>() -> anyhow::Result<Context<'a>> {
        Context::try_new()
    }

    /// Resolves `device`'s compute capability into a compilation [`Target`] -- overridable via the
    /// `TEENYC_CAPABILITY` env var (e.g. `sm_90`), mirroring
    /// `teeny_test::cuda::setup_cuda_env`'s behavior.
    pub fn default_target(device: &Device<'_>) -> anyhow::Result<Target> {
        let device_capability =
            teeny_cuda::compiler::target::capability_from_device_info(&device.info)?;

        let capability = if let Ok(val) = std::env::var("TEENYC_CAPABILITY") {
            val.strip_prefix("sm_")
                .and_then(|s| s.parse::<i32>().ok())
                .and_then(|n| {
                    teeny_cuda::compiler::target::Capability::from_major_minor(n / 10, n % 10)
                })
                .ok_or_else(|| {
                    anyhow::anyhow!("TEENYC_CAPABILITY={val:?} is not a recognised sm version")
                })?
        } else {
            device_capability
        };

        Ok(Target::new(capability))
    }

    /// A fixed [`Target`] for compile-only checks (e.g. snapshot tests) that don't need -- and
    /// shouldn't require -- a real device to be present. Always `Capability::Sm89`.
    pub fn reference_target() -> Target {
        Target::new(teeny_cuda::compiler::target::Capability::Sm89)
    }

    /// This backend's name, for disambiguating snapshot/fixture names that vary by compiled
    /// target (e.g. `format!("relu_mlir_{BACKEND_NAME}")`).
    pub const BACKEND_NAME: &str = "cuda";

    /// Loads a [`compile_kernel`]-produced artifact at `path` (raw PTX text) into a ready-to-run
    /// [`Program`].
    pub fn load_program<K: Kernel>(path: &str) -> anyhow::Result<Program<'static, K>> {
        let ptx = std::fs::read(path)?;
        Program::<K>::try_from_ptx(&ptx)
    }

    /// Builds a launch config from `program`'s metadata and `n_elements`, mirroring
    /// `teeny_test::cuda::launch_config_from_program`.
    pub fn launch_config<K: Kernel>(n_elements: usize, program: &Program<'_, K>) -> LaunchConfig {
        let threads = program.threads_per_block().max(1);
        LaunchConfig {
            grid: [(n_elements as u32).div_ceil(threads), 1, 1],
            block: [threads, 1, 1],
            cluster: [program.num_ctas().max(1), 1, 1],
        }
    }

    /// Builds a launch config with a pre-computed grid and block/cluster from `program`'s
    /// metadata, mirroring `teeny_test::cuda::launch_config_with_grid`. Use when the grid (number
    /// of CTAs) is known independently of element count.
    pub fn launch_config_with_grid<K: Kernel>(
        grid_x: usize,
        program: &Program<'_, K>,
    ) -> LaunchConfig {
        let threads = program.threads_per_block().max(1);
        LaunchConfig {
            grid: [grid_x as u32, 1, 1],
            block: [threads, 1, 1],
            cluster: [program.num_ctas().max(1), 1, 1],
        }
    }

    /// Builds a launch config from an explicit grid/block/cluster, for call sites that compute
    /// their own launch geometry rather than deriving it from `n_elements` or program metadata.
    pub fn launch_config_custom(
        grid: [u32; 3],
        block: [u32; 3],
        cluster: [u32; 3],
    ) -> LaunchConfig {
        LaunchConfig {
            grid,
            block,
            cluster,
        }
    }
}

#[cfg(all(feature = "riscv", not(feature = "cuda")))]
mod backend {
    use teeny_core::device::program::Kernel;

    /// The compiled-in backend's [`teeny_core::device::context::Context`] type.
    pub type Context<'a> = teeny_riscv::device::context::Riscv<'a>;
    /// The compiled-in backend's [`teeny_core::device::Device`] type.
    pub type Device<'a> = teeny_riscv::device::RiscvDevice<'a>;
    /// The compiled-in backend's [`teeny_core::device::LaunchConfig`] type.
    pub type LaunchConfig = teeny_riscv::device::RiscvLaunchConfig;
    /// The compiled-in backend's [`teeny_core::compiler::Target`] type.
    pub type Target = teeny_riscv::compiler::target::Target;
    /// The compiled-in backend's [`teeny_core::device::program::Program`] type.
    pub type Program<'a, K> = teeny_riscv::device::program::RiscvProgram<'a, K>;

    /// Compiles `kernel` for `target` (see `teeny_riscv::compiler::compile_kernel`).
    pub use teeny_riscv::compiler::compile_kernel;

    pub(crate) fn context<'a>() -> anyhow::Result<Context<'a>> {
        Context::try_new()
    }

    /// Always [`teeny_riscv::compiler::target::Capability::GenericRvv1_0`] -- unlike CUDA there
    /// is no hardware to detect a capability from; `device` is accepted only for a signature
    /// matching the `cuda` backend's `default_target`.
    pub fn default_target(_device: &Device<'_>) -> anyhow::Result<Target> {
        Ok(Target::new(
            teeny_riscv::compiler::target::Capability::GenericRvv1_0,
        ))
    }

    /// A fixed [`Target`] for compile-only checks (e.g. snapshot tests) that don't need -- and
    /// shouldn't require -- a real device to be present. Always
    /// `Capability::GenericRvv1_0` -- same value `default_target` uses, since RISC-V has no
    /// hardware-detected capability to fall back from.
    pub fn reference_target() -> Target {
        Target::new(teeny_riscv::compiler::target::Capability::GenericRvv1_0)
    }

    /// This backend's name, for disambiguating snapshot/fixture names that vary by compiled
    /// target (e.g. `format!("relu_mlir_{BACKEND_NAME}")`).
    pub const BACKEND_NAME: &str = "riscv";

    /// Loads a [`compile_kernel`]-produced artifact (a RISC-V ELF shared library) at `path`.
    /// Only succeeds when actually running on RISC-V (native, or under `qemu-riscv64`) -- see
    /// [`Program`]'s doc comment.
    pub fn load_program<K: Kernel>(path: &str) -> anyhow::Result<Program<'static, K>> {
        Program::<K>::try_new(path)
    }

    /// Always the empty [`LaunchConfig`] -- no real grid/block scheduling exists yet.
    pub fn launch_config<K: Kernel>(_n_elements: usize, _program: &Program<'_, K>) -> LaunchConfig {
        LaunchConfig::default()
    }

    /// Always the empty [`LaunchConfig`] -- see [`launch_config`].
    pub fn launch_config_with_grid<K: Kernel>(
        _grid_x: usize,
        _program: &Program<'_, K>,
    ) -> LaunchConfig {
        LaunchConfig::default()
    }

    /// Always the empty [`LaunchConfig`] -- see [`launch_config`].
    pub fn launch_config_custom(
        _grid: [u32; 3],
        _block: [u32; 3],
        _cluster: [u32; 3],
    ) -> LaunchConfig {
        LaunchConfig::default()
    }
}

pub use backend::{
    BACKEND_NAME, Context, Device, LaunchConfig, Program, Target, compile_kernel, default_target,
    launch_config, launch_config_custom, launch_config_with_grid, load_program, reference_target,
};

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
