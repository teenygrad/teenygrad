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

use std::path::PathBuf;

use dotenv::dotenv;
use insta::assert_debug_snapshot;
use teeny_core::device::Device;
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;

#[cfg(feature = "cuda")]
use teeny_cuda::compiler::{compile_kernel, target::Target};
#[cfg(feature = "cuda")]
use teeny_cuda::{compiler::target::Capability, errors::Result};
use teeny_test::load_fixture;

const N: usize = 1024;
const BLOCK_SIZE: i32 = 128;

/// CUDA-only: checks the compiled MLIR/PTX for a specific compute capability
/// (`Capability::Sm89`) -- inherently target-specific, not something to genericize.
#[test]
#[cfg(feature = "cuda")]
fn test_relu() -> Result<()> {
    dotenv().ok();

    let kernel = teeny_kernels::nn::activation::relu::ReluForward::<f32>::new(1024);
    let target = Target::new(Capability::Sm89);
    let ptx_path = PathBuf::from(compile_kernel(&kernel, &target, true, false)?);
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!("relu_source", kernel.source());
    assert_debug_snapshot!("relu_mlir", mlir.trim());

    Ok(())
}

/// Device-agnostic: `teeny_runtime::open()` resolves to whichever of `teeny-cuda`/`teeny-riscv`
/// is compiled in (see the `cuda`/`riscv` features), so this same test body runs against either
/// backend. On `riscv` it's expected to simply fail today -- `teeny_riscv::device::RiscvDevice`
/// can't yet load a compiled kernel on a non-RISC-V host (real hardware or `qemu-riscv64` would
/// be needed), and even then `Device::launch` is a stub until real per-kernel argument passing
/// lands (`teenygrad-1zd`). That's fine for now: this test exists to prove out the
/// `teeny-runtime` path, not to assert RISC-V correctness yet.
#[test]
fn test_relu_forward_gpu() -> anyhow::Result<()> {
    dotenv().ok();

    let device = teeny_runtime::open()?;
    let target = teeny_runtime::default_target(&device)?;

    let input_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "relu/x.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "relu/expected_forward.bin");
    let mut output_host = vec![0.0f32; N];

    let mut in_buf = device.buffer::<f32>(N)?;
    let out_buf = device.buffer::<f32>(N)?;
    in_buf.to_device(&input_host)?;

    let kernel = teeny_kernels::nn::activation::relu::ReluForward::<f32>::new(BLOCK_SIZE);
    let compiled_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::relu::ReluForward<f32>,
    >(&compiled_path)?;
    let cfg = teeny_runtime::launch_config(N, &program);

    let args = (
        in_buf.as_device_ptr() as *mut f32,
        out_buf.as_device_ptr() as *mut f32,
        N as i32,
    );

    device.launch(&program, &cfg, args)?;

    out_buf.to_host(&mut output_host)?;

    for i in 0..N {
        assert_eq!(
            output_host[i], expected[i],
            "relu mismatch at index {i}: input={}, actual={}, expected={}",
            input_host[i], output_host[i], expected[i]
        );
    }

    Ok(())
}

/// Device-agnostic -- see [`test_relu_forward_gpu`]'s doc comment.
#[test]
fn test_relu_backward_gpu() -> anyhow::Result<()> {
    dotenv().ok();

    let device = teeny_runtime::open()?;
    let target = teeny_runtime::default_target(&device)?;

    let y_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "relu/y_backward.bin");
    let dy_host = load_fixture(env!("CARGO_MANIFEST_DIR"), "relu/dy_backward.bin");
    let expected = load_fixture(env!("CARGO_MANIFEST_DIR"), "relu/expected_backward.bin");
    let mut dx_host = vec![0.0f32; N];

    let mut dy_buf = device.buffer::<f32>(N)?;
    let mut y_buf = device.buffer::<f32>(N)?;
    let dx_buf = device.buffer::<f32>(N)?;

    dy_buf.to_device(&dy_host)?;
    y_buf.to_device(&y_host)?;

    let kernel = teeny_kernels::nn::activation::relu::ReluBackward::<f32>::new(BLOCK_SIZE);
    let compiled_path = teeny_runtime::compile_kernel(&kernel, &target, true, false)?;
    let program = teeny_runtime::load_program::<
        teeny_kernels::nn::activation::relu::ReluBackward<f32>,
    >(&compiled_path)?;
    let cfg = teeny_runtime::launch_config(N, &program);

    let args = (
        dy_buf.as_device_ptr() as *mut f32,
        y_buf.as_device_ptr() as *mut f32,
        dx_buf.as_device_ptr() as *mut f32,
        N as i32,
    );

    device.launch(&program, &cfg, args)?;

    dx_buf.to_host(&mut dx_host)?;

    for i in 0..N {
        assert_eq!(
            dx_host[i], expected[i],
            "relu_backward mismatch at index {i}: y={}, dy={}, actual={}, expected={}",
            y_host[i], dy_host[i], dx_host[i], expected[i]
        );
    }

    Ok(())
}

#[test]
fn test_relu_forward_kernel_io_is_unary_elementwise() {
    use teeny_triton::PtrRole;

    let io = teeny_kernels::nn::activation::relu::ReluForward::<f32>::kernel_io();
    assert!(io.is_unary_elementwise());
    assert_eq!(io.roles, &[PtrRole::In, PtrRole::Out]);
    assert_eq!(io.n_in(), 1);
    assert_eq!(io.n_out(), 1);
    assert_eq!(io.n_inout(), 0);
    assert_eq!(io.n_unmarked(), 0);
    assert_eq!(io.first_in(), Some(0));
    assert_eq!(io.first_out(), Some(1));
}

#[test]
fn test_relu_forward_pointwise_fuse_probe() {
    use teeny_triton::{PointwiseFuseProbe, PointwiseFuseProbeExt};

    let k = teeny_kernels::nn::activation::relu::ReluForward::<f32>::new(1024);
    assert_eq!(
        k.pointwise_fuse_probe(),
        Some(PointwiseFuseProbe { block_size: 1024 })
    );
}

#[test]
fn test_softmax_forward_is_not_pointwise_fuse_probed() {
    // Softmax is unary In/Out with BLOCK_SIZE, but last arg is n_cols — the
    // macro must not stamp NElementsTiled, so the probe blanket does not apply.
    // SoftmaxForward therefore does not implement PointwiseFuseProbeExt.
    use teeny_triton::BlockSized;

    type Softmax = teeny_kernels::nn::activation::softmax::SoftmaxForward<f32>;
    assert!(Softmax::kernel_io().is_unary_elementwise());
    let k = Softmax::new(64);
    assert_eq!(k.block_size(), 64);
}
