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

use teeny_core::context::{Context, DeviceInfo, program::Kernel};

use crate::{
    context::Cuda,
    device::{CudaDevice, CudaLaunchConfig},
    errors::Result,
    program::CudaProgram,
    target::Capability,
};

pub struct CudaTestEnv {
    pub device: CudaDevice<'static>,
    pub capability: Capability,
}

pub fn setup_cuda_env() -> Result<CudaTestEnv> {
    let cuda_available = Cuda::is_available()?;
    assert!(cuda_available, "CUDA is not available");
    println!("[1/9] CUDA available");

    let cuda = Cuda::try_new()?;
    let devices = cuda.list_devices()?;
    assert!(!devices.is_empty(), "No CUDA devices found");
    println!("[2/9] found {} device(s)", devices.len());

    let device = cuda.device(&devices[0].id())?;
    let capability = Capability::from_device_info(&device.info)?;
    println!(
        "[3/9] device: {} (capability: {capability})",
        device.info.name
    );

    Ok(CudaTestEnv { device, capability })
}

pub fn launch_config(n_elements: usize, block_size: i32) -> CudaLaunchConfig {
    CudaLaunchConfig {
        grid: [(n_elements as u32).div_ceil(block_size as u32), 1, 1],
        block: [block_size as u32, 1, 1],
        cluster: [1, 1, 1],
    }
}

pub fn load_program_from_ptx<K: Kernel>(ptx: &[u8]) -> Result<CudaProgram<'static, K>> {
    println!("      loading PTX directly via driver JIT...");
    let program = CudaProgram::<K>::try_from_ptx(ptx, "entry_point")?;
    println!(
        "[7/9] loaded PTX: module={:#x} function={:#x}",
        program.module_ptr(),
        program.function_ptr(),
    );
    Ok(program)
}
