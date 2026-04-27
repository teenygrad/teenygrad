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
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_cuda::errors::Result;
use teeny_cuda::testing;

const N: usize = 1024;
const BLOCK_SIZE: i32 = 128;

#[test]
fn test_tensor_add() -> Result<()> {
    dotenv()?;
    let env = testing::setup_cuda_env()?;
    let device = env.device;

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
    let kernel = teeny_kernels::math::add::VectorAdd::<f32>::new(BLOCK_SIZE);
    let target = Target::new(env.capability);
    let ptx_path = compile_kernel(&kernel, &target, true)?;
    println!("[6/9] compiled PTX: {ptx_path}");
    let ptx = std::fs::read(&ptx_path)?;
    println!("      PTX size: {} bytes", ptx.len());
    println!(
        "──── PTX source ────\n{}\n────────────────────",
        String::from_utf8_lossy(&ptx)
    );

    let program = testing::load_program_from_ptx::<
        teeny_kernels::math::add::VectorAdd<f32>,
    >(&ptx)?;

    // ── Launch ─────────────────────────────────────────────────────────────
    let cfg = testing::launch_config(N, BLOCK_SIZE);
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
