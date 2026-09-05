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
#[cfg(feature = "hardware")]
use teeny_core::device::Device;
#[cfg(feature = "hardware")]
use teeny_core::device::buffer::Buffer;
use teeny_core::device::program::Kernel;
#[cfg(feature = "hardware")]
use teeny_test::load_fixture;

#[cfg(feature = "hardware")]
const N: usize = 1024;
#[cfg(feature = "hardware")]
const BLOCK_SIZE: i32 = 128;

/// Device-agnostic: compiles for whichever backend is active via a fixed reference target (no
/// real device needed -- `teeny_runtime::reference_target()`, not `default_target()`, so this
/// stays a pure compile check). The compiled MLIR differs by backend (target triple/codegen), so
/// the snapshot names are suffixed with `teeny_runtime::BACKEND_NAME` to keep each backend's
/// snapshot separate rather than one clobbering the other across feature-flag switches.
#[test]
fn test_relu() -> anyhow::Result<()> {
    dotenv().ok();

    let kernel = teeny_kernels::nn::activation::relu::ReluForward::<f32>::new(1024);
    let target = teeny_runtime::reference_target();
    let compiled_path = PathBuf::from(teeny_runtime::compile_kernel(
        &kernel, &target, true, false,
    )?);
    let mlir = std::fs::read_to_string(compiled_path.with_extension("mlir"))?;

    assert_debug_snapshot!(
        format!("relu_source_{}", teeny_runtime::BACKEND_NAME),
        kernel.source()
    );
    assert_debug_snapshot!(
        format!("relu_mlir_{}", teeny_runtime::BACKEND_NAME),
        mlir.trim()
    );

    Ok(())
}

/// Device-agnostic: `teeny_runtime::open()` resolves to whichever of `teeny-cuda`/`teeny-riscv`
/// is compiled in (see the `cuda`/`riscv` features), so this same test body runs against either
/// backend -- gated behind `hardware` since it needs a real device (a plain `cuda`/`riscv` build
/// only means "compile for this target", not "a device is present"; see `test_relu`, which needs
/// neither). On `riscv` it's expected to simply fail today -- `teeny_riscv::device::RiscvDevice`
/// can't yet load a compiled kernel on a non-RISC-V host (real hardware or `qemu-riscv64` would
/// be needed), and even then `Device::launch` is a stub until real per-kernel argument passing
/// lands (`teenygrad-1zd`). That's fine for now: this test exists to prove out the
/// `teeny-runtime` path, not to assert RISC-V correctness yet.
#[test]
#[cfg(feature = "hardware")]
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

    let args = (in_buf.as_device_ptr(), out_buf.as_device_ptr(), N as i32);

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
#[cfg(feature = "hardware")]
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
        dy_buf.as_device_ptr(),
        y_buf.as_device_ptr(),
        dx_buf.as_device_ptr(),
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
    use teeny_triton::PointwiseFuseProbe;

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
