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

//! Benchmarks comparing the three fused Conv2d+BatchNorm2d+SiLU kernel
//! variants (`conv2d_bn_silu`/scalar, `conv2d_bn_silu_tiled`, and
//! `conv2d_bn_silu_gemm`) across shapes chosen to straddle the shape-based
//! dispatch thresholds hand-picked in `graph/mod.rs` (`C_OUT >= 32` for GEMM
//! on 1x1 convs, `C_OUT >= 16` for the tiled kernel otherwise) — this gives
//! real data on whether those thresholds actually hold on the host GPU.
//!
//! On a Blackwell (sm_120) GPU, `teenyc`'s default PTX version for `sm_120a`
//! (.version 8.6) is rejected by at least some driver JIT compilers
//! ("PTX .version 8.6 does not support .target sm_120a"); if you hit that,
//! set `TEENYC_PTX_VERSION=87` (see `teeny-compiler`'s
//! `LlvmCompiler::with_ptx_version` / the `TEENYC_PTX_VERSION` env var) —
//! this is a `teenyc`-side default, not something this crate can fix.
//!
//! Run with:
//! ```bash
//! TEENYC_PTX_VERSION=87 cargo bench -p teeny-kernels --features cuda,training --bench conv2d_bn_silu
//! ```

use criterion::{Criterion, criterion_group, criterion_main};
use dotenv::dotenv;
use teeny_core::device::{Device, buffer::Buffer};
use teeny_cuda::compiler::{compile_kernel, target::Target};
use teeny_cuda::{device::CudaDevice, device::CudaLaunchConfig, testing};
use teeny_kernels::nn::fused::{
    conv2d_bn_silu::Conv2dBnSiluForward, conv2d_bn_silu_gemm::Conv2dBnSiluGemmForward,
    conv2d_bn_silu_tiled::Conv2dBnSiluTiledForward,
};

const BLOCK_OW_SCALAR: i32 = 16;
const BLOCK_OW_TILED: i32 = 16;
const BLOCK_N_TILED: i32 = 16;
const BLOCK_M_GEMM: i32 = 32;
const BLOCK_N_GEMM: i32 = 32;
const BLOCK_K_GEMM: i32 = 32;
const GROUP_M_GEMM: i32 = 8;

/// Shared conv shape parameters for one benchmark scenario.
struct ConvShape {
    label: &'static str,
    nb: usize,
    c_in: usize,
    c_out: usize,
    hh: usize,
    ww: usize,
    kh: i32,
    kw: i32,
    stride_h: i32,
    stride_w: i32,
    pad_h: i32,
    pad_w: i32,
}

impl ConvShape {
    fn oh(&self) -> usize {
        (self.hh + 2 * self.pad_h as usize - self.kh as usize) / self.stride_h as usize + 1
    }

    fn ow(&self) -> usize {
        (self.ww + 2 * self.pad_w as usize - self.kw as usize) / self.stride_w as usize + 1
    }

    fn x_host(&self) -> Vec<f32> {
        (0..self.nb * self.c_in * self.hh * self.ww)
            .map(|i| (i as f32 % 17.0 - 8.0) * 0.1)
            .collect()
    }

    fn bn_scale(&self) -> Vec<f32> {
        (0..self.c_out).map(|i| 0.8 + i as f32 * 0.05).collect()
    }

    fn bn_shift(&self) -> Vec<f32> {
        (0..self.c_out).map(|i| i as f32 * 0.1 - 0.15).collect()
    }
}

/// Benchmarks the scalar/direct kernel (`Conv2dBnSiluForward`) — legal for any shape.
fn bench_scalar(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    device: &CudaDevice<'static>,
    target: &Target,
    shape: &ConvShape,
) -> anyhow::Result<()> {
    let (oh, ow) = (shape.oh(), shape.ow());
    let w_host: Vec<f32> = (0..shape.c_out * shape.c_in * shape.kh as usize * shape.kw as usize)
        .map(|i| (i as f32 % 13.0 - 6.0) * 0.05)
        .collect();

    let mut x_buf = device.buffer::<f32>(shape.nb * shape.c_in * shape.hh * shape.ww)?;
    let mut w_buf = device.buffer::<f32>(w_host.len())?;
    let mut s_buf = device.buffer::<f32>(shape.c_out)?;
    let mut sh_buf = device.buffer::<f32>(shape.c_out)?;
    let y_buf = device.buffer::<f32>(shape.nb * shape.c_out * oh * ow)?;

    x_buf.to_device(&shape.x_host())?;
    w_buf.to_device(&w_host)?;
    s_buf.to_device(&shape.bn_scale())?;
    sh_buf.to_device(&shape.bn_shift())?;

    let kernel = Conv2dBnSiluForward::new(
        shape.kh,
        shape.kw,
        shape.stride_h,
        shape.stride_w,
        shape.pad_h,
        shape.pad_w,
        1,
        BLOCK_OW_SCALAR,
    );
    let ptx = std::fs::read(compile_kernel(&kernel, target, false, false)?)?;
    let program = testing::load_program_from_ptx::<Conv2dBnSiluForward>(&ptx)?;

    let num_ow_tiles = ow.div_ceil(BLOCK_OW_SCALAR as usize);
    let grid = (shape.nb * shape.c_out * oh * num_ow_tiles) as u32;
    let cfg = CudaLaunchConfig {
        grid: [grid, 1, 1],
        block: [128, 1, 1],
        cluster: [1, 1, 1],
    };

    group.bench_function(format!("scalar/{}", shape.label), |b| {
        b.iter(|| {
            device
                .launch(
                    &program,
                    &cfg,
                    (
                        x_buf.as_device_ptr() as *mut f32,
                        w_buf.as_device_ptr() as *mut f32,
                        s_buf.as_device_ptr() as *mut f32,
                        sh_buf.as_device_ptr() as *mut f32,
                        y_buf.as_device_ptr() as *mut f32,
                        shape.nb as i32,
                        shape.c_in as i32,
                        shape.c_out as i32,
                        shape.hh as i32,
                        shape.ww as i32,
                        oh as i32,
                        ow as i32,
                    ),
                )
                .unwrap();
        });
    });

    Ok(())
}

/// Benchmarks the channel-tiled kernel (`Conv2dBnSiluTiledForward`) — legal whenever groups==1.
fn bench_tiled(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    device: &CudaDevice<'static>,
    target: &Target,
    shape: &ConvShape,
) -> anyhow::Result<()> {
    let (oh, ow) = (shape.oh(), shape.ow());
    let w_host: Vec<f32> = (0..shape.c_out * shape.c_in * shape.kh as usize * shape.kw as usize)
        .map(|i| (i as f32 % 13.0 - 6.0) * 0.05)
        .collect();
    // Matches the padding rule in test_conv2d_bn_silu_tiled.rs: avoids TMA overlap/misalignment.
    let y_col_stride = ow.max(BLOCK_OW_TILED as usize).next_multiple_of(4);
    let y_total = shape.nb * shape.c_out * oh * y_col_stride;

    let mut x_buf = device.buffer::<f32>(shape.nb * shape.c_in * shape.hh * shape.ww)?;
    let mut w_buf = device.buffer::<f32>(w_host.len())?;
    let mut s_buf = device.buffer::<f32>(shape.c_out)?;
    let mut sh_buf = device.buffer::<f32>(shape.c_out)?;
    let y_buf = device.buffer::<f32>(y_total)?;

    x_buf.to_device(&shape.x_host())?;
    w_buf.to_device(&w_host)?;
    s_buf.to_device(&shape.bn_scale())?;
    sh_buf.to_device(&shape.bn_shift())?;

    let kernel = Conv2dBnSiluTiledForward::new(
        shape.kh,
        shape.kw,
        shape.stride_h,
        shape.stride_w,
        shape.pad_h,
        shape.pad_w,
        BLOCK_OW_TILED,
        BLOCK_N_TILED,
    );
    let ptx = std::fs::read(compile_kernel(&kernel, target, false, false)?)?;
    let program = testing::load_program_from_ptx::<Conv2dBnSiluTiledForward>(&ptx)?;

    let num_ow_tiles = ow.div_ceil(BLOCK_OW_TILED as usize);
    let num_n_tiles = shape.c_out.div_ceil(BLOCK_N_TILED as usize);
    let grid = (shape.nb * oh * num_n_tiles * num_ow_tiles) as u32;
    let cfg = CudaLaunchConfig {
        grid: [grid, 1, 1],
        block: [128, 1, 1],
        cluster: [1, 1, 1],
    };

    group.bench_function(format!("tiled/{}", shape.label), |b| {
        b.iter(|| {
            device
                .launch(
                    &program,
                    &cfg,
                    (
                        x_buf.as_device_ptr() as *mut f32,
                        w_buf.as_device_ptr() as *mut f32,
                        s_buf.as_device_ptr() as *mut f32,
                        sh_buf.as_device_ptr() as *mut f32,
                        y_buf.as_device_ptr() as *mut f32,
                        shape.nb as i32,
                        shape.c_in as i32,
                        shape.c_out as i32,
                        shape.hh as i32,
                        shape.ww as i32,
                        oh as i32,
                        ow as i32,
                        y_col_stride as i32,
                    ),
                )
                .unwrap();
        });
    });

    Ok(())
}

/// Benchmarks the GEMM/tensor-core kernel (`Conv2dBnSiluGemmForward`) — only legal for
/// 1x1/stride=1/pad=0/groups=1 convs.
fn bench_gemm(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    device: &CudaDevice<'static>,
    target: &Target,
    shape: &ConvShape,
) -> anyhow::Result<()> {
    assert!(
        shape.kh == 1 && shape.kw == 1 && shape.stride_h == 1 && shape.stride_w == 1,
        "GEMM variant only supports 1x1/stride=1/pad=0 convs"
    );
    let m = shape.hh * shape.ww;
    let w_host: Vec<f32> = (0..shape.c_out * shape.c_in)
        .map(|i| (i as f32 % 13.0 - 6.0) * 0.05)
        .collect();

    let mut x_buf = device.buffer::<f32>(shape.nb * shape.c_in * shape.hh * shape.ww)?;
    let mut w_buf = device.buffer::<f32>(w_host.len())?;
    let mut s_buf = device.buffer::<f32>(shape.c_out)?;
    let mut sh_buf = device.buffer::<f32>(shape.c_out)?;
    let y_buf = device.buffer::<f32>(shape.nb * shape.c_out * m)?;

    x_buf.to_device(&shape.x_host())?;
    w_buf.to_device(&w_host)?;
    s_buf.to_device(&shape.bn_scale())?;
    sh_buf.to_device(&shape.bn_shift())?;

    let kernel =
        Conv2dBnSiluGemmForward::new(BLOCK_M_GEMM, BLOCK_N_GEMM, BLOCK_K_GEMM, GROUP_M_GEMM);
    let ptx = std::fs::read(compile_kernel(&kernel, target, false, false)?)?;
    let program = testing::load_program_from_ptx::<Conv2dBnSiluGemmForward>(&ptx)?;

    let num_pm = m.div_ceil(BLOCK_M_GEMM as usize);
    let num_pn = shape.c_out.div_ceil(BLOCK_N_GEMM as usize);
    let grid = (shape.nb * num_pm * num_pn) as u32;
    let cfg = CudaLaunchConfig {
        grid: [grid, 1, 1],
        block: [128, 1, 1],
        cluster: [1, 1, 1],
    };

    group.bench_function(format!("gemm/{}", shape.label), |b| {
        b.iter(|| {
            device
                .launch(
                    &program,
                    &cfg,
                    (
                        x_buf.as_device_ptr() as *mut f32,
                        w_buf.as_device_ptr() as *mut f32,
                        s_buf.as_device_ptr() as *mut f32,
                        sh_buf.as_device_ptr() as *mut f32,
                        y_buf.as_device_ptr() as *mut f32,
                        shape.nb as i32,
                        shape.c_in as i32,
                        shape.c_out as i32,
                        m as i32,
                    ),
                )
                .unwrap();
        });
    });

    Ok(())
}

fn bench_conv2d_bn_silu(c: &mut Criterion) {
    dotenv().ok();
    let env = testing::setup_cuda_env().expect("CUDA setup failed");
    let device = &env.device;
    let target = Target::new(env.capability);

    // Right at the GEMM threshold (C_OUT>=32, 1x1 conv): the exact shape from
    // test_conv2d_bn_silu_gemm_c_out32_c_in64_m1600 ("model.6.m.0.cv2" layer config).
    // All 3 variants are legal here — this is the interesting comparison.
    let gemm_threshold = ConvShape {
        label: "1x1_c_out32_c_in64_40x40",
        nb: 1,
        c_in: 64,
        c_out: 32,
        hh: 40,
        ww: 40,
        kh: 1,
        kw: 1,
        stride_h: 1,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
    };
    {
        let mut group = c.benchmark_group("conv2d_bn_silu/gemm_threshold");
        bench_scalar(&mut group, device, &target, &gemm_threshold).unwrap();
        bench_tiled(&mut group, device, &target, &gemm_threshold).unwrap();
        bench_gemm(&mut group, device, &target, &gemm_threshold).unwrap();
        group.finish();
    }

    // At the tiled threshold (C_OUT==16), below the GEMM one: dispatch picks tiled, but
    // include GEMM too (still legal for a 1x1 conv) to see how early its crossover really is.
    let tiled_threshold = ConvShape {
        label: "1x1_c_out16_c_in64_32x32",
        nb: 1,
        c_in: 64,
        c_out: 16,
        hh: 32,
        ww: 32,
        kh: 1,
        kw: 1,
        stride_h: 1,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
    };
    {
        let mut group = c.benchmark_group("conv2d_bn_silu/tiled_threshold");
        bench_scalar(&mut group, device, &target, &tiled_threshold).unwrap();
        bench_tiled(&mut group, device, &target, &tiled_threshold).unwrap();
        bench_gemm(&mut group, device, &target, &tiled_threshold).unwrap();
        group.finish();
    }

    // Below both thresholds (C_OUT==8): dispatch picks scalar; again include tiled/gemm
    // anyway to map out the actual crossover points.
    let below_threshold = ConvShape {
        label: "1x1_c_out8_c_in64_32x32",
        nb: 1,
        c_in: 64,
        c_out: 8,
        hh: 32,
        ww: 32,
        kh: 1,
        kw: 1,
        stride_h: 1,
        stride_w: 1,
        pad_h: 0,
        pad_w: 0,
    };
    {
        let mut group = c.benchmark_group("conv2d_bn_silu/below_threshold");
        bench_scalar(&mut group, device, &target, &below_threshold).unwrap();
        bench_tiled(&mut group, device, &target, &below_threshold).unwrap();
        bench_gemm(&mut group, device, &target, &below_threshold).unwrap();
        group.finish();
    }

    // 3x3 conv, C_OUT=32: GEMM is not legal here (kernel != 1x1) — scalar vs tiled only.
    let non_1x1 = ConvShape {
        label: "3x3_c_out32_c_in64_40x40",
        nb: 1,
        c_in: 64,
        c_out: 32,
        hh: 40,
        ww: 40,
        kh: 3,
        kw: 3,
        stride_h: 1,
        stride_w: 1,
        pad_h: 1,
        pad_w: 1,
    };
    {
        let mut group = c.benchmark_group("conv2d_bn_silu/non_1x1");
        bench_scalar(&mut group, device, &target, &non_1x1).unwrap();
        bench_tiled(&mut group, device, &target, &non_1x1).unwrap();
        group.finish();
    }
}

criterion_group!(benches, bench_conv2d_bn_silu);
criterion_main!(benches);
