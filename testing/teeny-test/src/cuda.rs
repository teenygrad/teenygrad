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

use teeny_core::device::{
    context::{Context, DeviceInfo},
    program::Kernel,
};

use teeny_cuda::{
    compiler::target::{Capability, capability_from_device_info},
    device::context::Cuda,
    device::program::CudaProgram,
    device::{CudaDevice, CudaLaunchConfig},
    errors::Result,
};

/// A device + capability pair set up for a test, via [`setup_cuda_env`].
pub struct CudaTestEnv {
    /// The opened test device.
    pub device: CudaDevice<'static>,
    /// The device's (possibly `TEENYC_CAPABILITY`-overridden) compute capability.
    pub capability: Capability,
}

/// Asserts CUDA is available, opens the first device, and resolves its compute capability
/// (overridable via the `TEENYC_CAPABILITY` env var, e.g. `sm_90`) — the standard setup for
/// any CUDA-backed test.
pub fn setup_cuda_env() -> Result<CudaTestEnv> {
    let cuda_available = Cuda::is_available()?;
    assert!(cuda_available, "CUDA is not available");
    println!("[1/9] CUDA available");

    let cuda = Cuda::try_new()?;
    let devices = cuda.list_devices()?;
    assert!(!devices.is_empty(), "No CUDA devices found");
    println!("[2/9] found {} device(s)", devices.len());

    let device = cuda.device(&devices[0].id())?;
    let device_capability = capability_from_device_info(&device.info)?;

    let capability = if let Ok(val) = std::env::var("TEENYC_CAPABILITY") {
        let parsed = val
            .strip_prefix("sm_")
            .and_then(|s| s.parse::<i32>().ok())
            .and_then(|n| Capability::from_major_minor(n / 10, n % 10))
            .ok_or_else(|| {
                teeny_cuda::errors::Error::UnknownCapability(format!(
                    "TEENYC_CAPABILITY={val:?} is not a recognised sm version"
                ))
            })?;
        println!(
            "[3/9] device: {} (capability overridden: {device_capability} → {parsed})",
            device.info.name
        );
        parsed
    } else {
        println!(
            "[3/9] device: {} (capability: {device_capability})",
            device.info.name
        );
        device_capability
    };

    Ok(CudaTestEnv { device, capability })
}

/// Build a launch config using block size from kernel metadata.
///
/// `n_elements` — total number of elements to process.
/// `program`    — compiled program; `metadata.threads_per_block()` is used as the block size.
pub fn launch_config_from_program<K: Kernel>(
    n_elements: usize,
    program: &CudaProgram<'_, K>,
) -> CudaLaunchConfig {
    let threads = program.threads_per_block().max(1);
    CudaLaunchConfig {
        grid: [(n_elements as u32).div_ceil(threads), 1, 1],
        block: [threads, 1, 1],
        cluster: [program.num_ctas().max(1), 1, 1],
    }
}

/// Build a launch config with an explicit block size.
///
/// Use when the block size is determined at the call site rather than from PTX metadata
/// (e.g. when the grid is computed before the program is compiled).
pub fn launch_config(n_elements: usize, block_size: i32) -> CudaLaunchConfig {
    CudaLaunchConfig {
        grid: [(n_elements as u32).div_ceil(block_size as u32), 1, 1],
        block: [block_size as u32, 1, 1],
        cluster: [1, 1, 1],
    }
}

/// Build a launch config with a pre-computed grid and block from kernel metadata.
///
/// Use when the launch grid (number of CTAs) is known independently of element count —
/// for example, tiled matmul kernels where grid = `ceil(M/TILE_M) * ceil(N/TILE_N)`.
/// The block and cluster dimensions are read from `program.metadata`.
pub fn launch_config_with_grid<K: Kernel>(
    grid_x: usize,
    program: &CudaProgram<'_, K>,
) -> CudaLaunchConfig {
    let threads = program.threads_per_block().max(1);
    CudaLaunchConfig {
        grid: [grid_x as u32, 1, 1],
        block: [threads, 1, 1],
        cluster: [program.num_ctas().max(1), 1, 1],
    }
}

/// Loads a compiled program directly from raw PTX bytes via the driver's JIT loader (skipping
/// `nvptxcompiler`) — useful in tests that already have PTX in hand.
pub fn load_program_from_ptx<K: Kernel>(ptx: &[u8]) -> Result<CudaProgram<'static, K>> {
    println!("      loading PTX directly via driver JIT...");
    let program = CudaProgram::<K>::try_from_ptx(ptx)?;
    println!(
        "[7/9] loaded PTX: module={:#x} function={:#x} num_warps={} num_ctas={}",
        program.module_ptr(),
        program.function_ptr(),
        program.num_warps(),
        program.num_ctas(),
    );
    Ok(program)
}
