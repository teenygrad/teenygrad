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

use dotenv::dotenv;

use teeny_compiler::compiler::driver::cuda::compile_kernel;
use teeny_compiler::compiler::target::cuda::Target;
use teeny_core::context::Context;
use teeny_core::context::buffer::Buffer;
use teeny_core::context::device::Device;
use teeny_cuda::context::Cuda;
use teeny_cuda::device::CudaLaunchConfig;
use teeny_cuda::errors::Result;
use teeny_cuda::target::Capability;

const N: usize = 1024;
const BLOCK_SIZE: i32 = 128;

#[test]
fn test_tensor_add() -> Result<()> {
    dotenv()?;

    let cuda_available = Cuda::is_available()?;
    assert!(cuda_available, "CUDA is not available");
    println!("[1/9] CUDA available");

    let cuda = Cuda::try_new()?;
    let devices = cuda.list_devices()?;
    assert!(!devices.is_empty(), "No CUDA devices found");
    println!("[2/9] found {} device(s)", devices.len());

    let device = cuda.device(&devices[0].id)?;
    // let capability = Capability::from_device_info(&device.info)?;
    let capability = Capability::Sm90;
    println!(
        "[3/9] device: {} (capability: {capability})",
        device.info.name
    );

    // ── Host data ──────────────────────────────────────────────────────────
    let x_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let y_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();
    let mut output_host = vec![0.0f32; N];

    // ── Device buffers ─────────────────────────────────────────────────────
    let mut x_buf = device.buffer::<f32>(N)?;
    let mut y_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;
    println!(
        "[4/9] allocated device buffers: x={:#x} y={:#x} out={:#x}",
        x_buf.as_device_ptr(),
        y_buf.as_device_ptr(),
        out_buf.as_device_ptr(),
    );

    x_buf.to_device(&x_host)?;
    y_buf.to_device(&y_host)?;
    println!("[5/9] copied input data to device");

    // ── Compile PTX → cubin → CudaProgram ─────────────────────────────────
    let kernel = teeny_kernels::math::add::VectorAdd::<f32, BLOCK_SIZE>::new();
    let target = Target::new(capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    println!("[6/9] compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;
    println!("      PTX size: {} bytes", ptx.len());
    println!(
        "──── PTX source ────\n{}\n────────────────────",
        String::from_utf8_lossy(&ptx)
    );

    println!("      loading PTX directly via driver JIT...");
    let program = teeny_cuda::program::CudaProgram::<
        teeny_kernels::math::add::VectorAdd<f32, BLOCK_SIZE>,
    >::try_from_ptx(&ptx, "entry_point")?;
    println!(
        "[7/9] loaded cubin: module={:#x} function={:#x}",
        program.module_ptr(),
        program.function_ptr(),
    );

    // ── Launch ─────────────────────────────────────────────────────────────
    let cfg = CudaLaunchConfig {
        grid: [(N as u32).div_ceil(BLOCK_SIZE as u32), 1, 1],
        block: [BLOCK_SIZE as u32, 1, 1],
        cluster: [1, 1, 1],
    };
    println!(
        "[8/9] launching: grid={:?} block={:?} n_elements={N}",
        cfg.grid, cfg.block,
    );

    let args = (
        x_buf.as_device_ptr() as *mut f32,
        y_buf.as_device_ptr() as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        N as i32,
    );

    println!("Launching with args: {:?}", args);

    device.launch(&program, &cfg, args)?;
    println!("      kernel completed (synchronized)");

    // ── Copy results back and verify ───────────────────────────────────────
    out_buf.to_host(&mut output_host)?;
    println!(
        "[9/9] copied results back: output[0]={} output[1023]={}",
        output_host[0], output_host[1023]
    );

    for i in 0..N {
        assert_eq!(
            output_host[i],
            x_host[i] + y_host[i],
            "mismatch at index {i}"
        );
    }

    Ok(())
}
