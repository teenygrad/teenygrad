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

//! `MemTraffic`/`CostModel` calibration against real CUDA hardware
//! (teenygrad-3w0.4).
//!
//! `device.launch(...)` (`drivers/teeny-cuda/src/device/mod.rs`) synchronizes
//! internally (`cuCtxSynchronize`) before returning, so wall-clock timing
//! around repeated launches genuinely measures GPU execution time, not just
//! host-side dispatch — the same approach every other timing-sensitive test
//! in this crate would use if it needed to (none currently do; this is the
//! first). No new CUDA-driver bindings needed: everything here goes through
//! the existing public `teeny_cuda::device`/`testing` API.
//!
//! These are empirical, hardware-specific measurements, not universal
//! constants — see `CostModel`'s doc comment. Re-run and re-derive the
//! constants below if this ever needs to target different hardware.

#![cfg(feature = "cuda")]

use std::time::Instant;

use dotenv::dotenv;
use teeny_core::device::Device;
use teeny_cuda::compiler::{compile_kernel, target::Target};
use teeny_cuda::device::CudaDevice;
use teeny_cuda::errors::Result;
use teeny_cuda::testing;
use teeny_kernels::nn::activation::elu::EluForward;
use teeny_kernels::nn::conv::conv2d::Conv2dForward;
use teeny_triton::{CostModel, mem_traffic};

const WARMUP_ITERS: usize = 3;
const TIMED_ITERS: usize = 15;

/// Median wall-clock seconds for `TIMED_ITERS` calls to `launch`, after
/// `WARMUP_ITERS` untimed warm-up calls (first-launch JIT/caching effects).
fn median_launch_seconds(mut launch: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..WARMUP_ITERS {
        launch()?;
    }
    let mut samples = Vec::with_capacity(TIMED_ITERS);
    for _ in 0..TIMED_ITERS {
        let start = Instant::now();
        launch()?;
        samples.push(start.elapsed().as_secs_f64());
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(samples[samples.len() / 2])
}

fn achieved_gb_per_s(bytes: u64, seconds: f64) -> f64 {
    (bytes as f64) / seconds / 1e9
}

/// `elu_forward` at a fixed BLOCK_SIZE, varying only `n_elements` (so only
/// CTA count / occupancy changes between the two launches, not per-CTA
/// kernel shape) — isolates the under-parallelism effect from everything
/// else `mem_traffic`/`CostModel` might otherwise conflate it with.
fn elu_median_seconds_and_bytes(
    device: &CudaDevice<'_>,
    capability: teeny_cuda::compiler::target::Capability,
    block_size: i32,
    n_elements: usize,
) -> Result<(f64, u64)> {
    let x_buf = device.buffer::<f32>(n_elements)?;
    let y_buf = device.buffer::<f32>(n_elements)?;

    let kernel = EluForward::<f32>::new(block_size);
    let ptx = std::fs::read(compile_kernel(
        &kernel,
        &Target::new(capability),
        true,
        false,
    )?)?;
    let program = testing::load_program_from_ptx::<EluForward<f32>>(&ptx)?;
    let cfg = testing::launch_config_from_program(n_elements, &program);

    let seconds = median_launch_seconds(|| {
        device.launch(
            &program,
            &cfg,
            (
                x_buf.as_device_ptr() as *mut f32,
                y_buf.as_device_ptr() as *mut f32,
                n_elements as i32,
                1.0_f32,
            ),
        )
    })?;

    let spec = EluForward::<f32>::tile_spec();
    let resolve = |name: &str| match name {
        "BLOCK_SIZE" => block_size as i64,
        "n_elements" => n_elements as i64,
        other => panic!("unexpected param {other}"),
    };
    let bytes = mem_traffic(&spec, size_of::<f32>(), resolve);
    Ok((seconds, bytes))
}

/// Confirms the direction `CostModel::under_parallel_penalty` assumes: an
/// `elu_forward` launch with too few CTAs to keep every SM busy achieves
/// lower bandwidth than the same kernel comfortably oversubscribed, even
/// though both move bytes at the same per-CTA rate in `mem_traffic`'s model.
#[test]
fn calibration_under_parallel_launch_achieves_lower_bandwidth() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let sm_count = env.device.info.multi_processor_count as usize;
    let block_size = 1024i32;

    // ~1 CTA/SM: minimal occupancy, can't hide memory latency behind other
    // warps' work. ~32 CTAs/SM: comfortably oversubscribed.
    let small_n = sm_count * block_size as usize;
    let large_n = sm_count * block_size as usize * 32;

    let (small_s, small_bytes) =
        elu_median_seconds_and_bytes(&env.device, env.capability, block_size, small_n)?;
    let (large_s, large_bytes) =
        elu_median_seconds_and_bytes(&env.device, env.capability, block_size, large_n)?;

    let small_gbps = achieved_gb_per_s(small_bytes, small_s);
    let large_gbps = achieved_gb_per_s(large_bytes, large_s);
    eprintln!(
        "elu_forward: {small_n} elements (~1 CTA/SM) = {small_gbps:.1} GB/s, \
         {large_n} elements (~32 CTA/SM) = {large_gbps:.1} GB/s, ratio = {:.2}x",
        large_gbps / small_gbps
    );

    assert!(
        large_gbps > small_gbps,
        "expected the oversubscribed launch ({large_gbps:.1} GB/s) to beat the \
         under-parallel one ({small_gbps:.1} GB/s) — if this fails, either this \
         GPU doesn't show the effect CostModel::under_parallel_penalty assumes, \
         or the chosen small_n/large_n aren't a clean enough isolation of it"
    );
    Ok(())
}

/// Compares `conv2d_forward`'s achieved bandwidth (whose `x_ptr` reads a
/// wider, `TileWindow`-adjusted footprint per `mem_traffic`) against
/// `elu_forward`'s (plain contiguous access), both comfortably parallel.
///
/// **Originally discovered while writing this test, now fixed**:
/// `mem_traffic` used to only sum the axes actually present in a tensor's
/// declared `axes` list — `conv2d_forward`'s `x_ptr`/`y_ptr` tile specs
/// (teenygrad-3w0.5) only declare the `W`/`OW` axis, so `B`/`C_IN`/`C_OUT`/
/// `H`/`OH` were silently omitted, undercounting real traffic by ~4-5 orders
/// of magnitude on this hardware. Fixed in teenygrad-3w0.8 via
/// `TensorTileSpec::untiled_dims`, applied to `conv2d_forward` — this test
/// now reports the corrected `mem_traffic`-derived GB/s alongside the
/// original per-output-element timing (kept because it's a still-useful,
/// complementary signal: `mem_traffic` doesn't model compute intensity, so
/// `conv2d_forward`'s much higher compute-per-element means its bandwidth
/// figure alone still isn't a clean coalescing-only isolation — see
/// `CostModel::window_penalty`'s doc comment for why that field stays an
/// uncalibrated placeholder even after this fix).
#[test]
fn calibration_windowed_kernel_is_slower_per_output_element() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let sm_count = env.device.info.multi_processor_count as usize;
    let block_size = 1024i32;
    let large_n = sm_count * block_size as usize * 32;
    let (elu_s, _elu_bytes) =
        elu_median_seconds_and_bytes(&env.device, env.capability, block_size, large_n)?;
    let elu_s_per_elem = elu_s / large_n as f64;

    // A conv2d_forward shape large enough for a meaningful timing signal.
    let b = 4usize;
    let c_in = 16usize;
    let c_out = 32usize;
    let hh = 64usize;
    let ww = 64usize;
    let kh = 3i32;
    let kw = 3i32;
    let stride = 1i32;
    let pad = 1i32;
    let block_ow = 32i32;
    let oh = (hh as i32 + 2 * pad - kh) / stride + 1;
    let ow = (ww as i32 + 2 * pad - kw) / stride + 1;

    let x_buf = env.device.buffer::<f32>(b * c_in * hh * ww)?;
    let w_buf = env
        .device
        .buffer::<f32>(c_out * c_in * kh as usize * kw as usize)?;
    let y_buf = env
        .device
        .buffer::<f32>(b * c_out * oh as usize * ow as usize)?;

    let kernel = Conv2dForward::<f32>::new(kh, kw, stride, stride, pad, pad, 1, block_ow);
    let ptx = std::fs::read(compile_kernel(
        &kernel,
        &Target::new(env.capability),
        true,
        false,
    )?)?;
    let program = testing::load_program_from_ptx::<Conv2dForward<f32>>(&ptx)?;
    let num_ow_tiles = (ow as usize).div_ceil(block_ow as usize);
    let grid_x = b * c_out * oh as usize * num_ow_tiles;
    let cfg = testing::launch_config_with_grid::<Conv2dForward<f32>>(grid_x, &program);

    let conv_s = median_launch_seconds(|| {
        env.device.launch(
            &program,
            &cfg,
            (
                x_buf.as_device_ptr() as *mut f32,
                w_buf.as_device_ptr() as *mut f32,
                y_buf.as_device_ptr() as *mut f32,
                b as i32,
                c_in as i32,
                c_out as i32,
                hh as i32,
                ww as i32,
                oh,
                ow,
            ),
        )
    })?;
    // Now that untiled_dims (teenygrad-3w0.8) is fixed, mem_traffic's byte
    // count for conv2d_forward is a real order-of-magnitude estimate.
    let spec = Conv2dForward::<f32>::tile_spec();
    let resolve = |name: &str| match name {
        "BLOCK_OW" => block_ow as i64,
        "OW" => ow as i64,
        "W" => ww as i64,
        "STRIDE_W" => stride as i64,
        "PAD_W" => pad as i64,
        "KW" => kw as i64,
        "_B" => b as i64,
        "C_IN" => c_in as i64,
        "C_OUT" => c_out as i64,
        "H" => hh as i64,
        "OH" => oh as i64,
        other => panic!("unexpected param {other}"),
    };
    let conv_bytes = mem_traffic(&spec, size_of::<f32>(), resolve);
    let conv_gbps = achieved_gb_per_s(conv_bytes, conv_s);

    let n_outputs = b * c_out * oh as usize * ow as usize;
    let conv_s_per_elem = conv_s / n_outputs as f64;

    eprintln!(
        "elu_forward = {elu_s_per_elem:.3e} s/elem, conv2d_forward = \
         {conv_s_per_elem:.3e} s/elem, ratio = {:.2}x | conv2d_forward \
         mem_traffic-derived bandwidth = {conv_gbps:.1} GB/s (sanity: should be a \
         plausible GPU bandwidth figure, not near-zero like before the \
         untiled_dims fix)",
        conv_s_per_elem / elu_s_per_elem
    );

    assert!(
        conv_s_per_elem > elu_s_per_elem,
        "expected conv2d_forward ({conv_s_per_elem:.3e} s/elem) to be slower per \
         output element than elu_forward's plain contiguous access \
         ({elu_s_per_elem:.3e} s/elem) — if this fails, revisit whether \
         CostModel::window_penalty's premise holds on this hardware"
    );
    assert!(
        conv_gbps > 1.0,
        "conv2d_forward's mem_traffic-derived bandwidth ({conv_gbps:.4} GB/s) is \
         implausibly low — the untiled_dims fix may have regressed"
    );
    Ok(())
}

/// End-to-end check: a `CostModel` built from this hardware's real SM count
/// and conservative, rounded-down versions of the ratios measured above
/// ranks a handful of real (kernel, config) launches in the same order as
/// their actual measured latency.
#[test]
fn cost_model_ranks_real_launches_like_measured_latency() -> Result<()> {
    dotenv().ok();
    let env = testing::setup_cuda_env()?;
    let sm_count = env.device.info.multi_processor_count as usize;
    let block_size = 1024i32;
    let small_n = sm_count * block_size as usize;
    let large_n = sm_count * block_size as usize * 32;

    let (small_s, _) =
        elu_median_seconds_and_bytes(&env.device, env.capability, block_size, small_n)?;
    let (large_s, _) =
        elu_median_seconds_and_bytes(&env.device, env.capability, block_size, large_n)?;

    // Conservative constants: real measured ratios (see the two tests above)
    // are typically well above these on this hardware; picking a smaller
    // value here just means the model under-claims the effect rather than
    // over-claiming it if hardware/driver conditions vary between runs.
    let cost_model = CostModel {
        sm_count: sm_count as u32,
        saturation_ctas_per_sm: 8,
        under_parallel_penalty: 1.5,
        window_penalty: 1.2,
        // Not exercised by this test (elu_forward doesn't touch shared
        // memory) -- real hardware value for consistency, neutral penalty.
        shared_mem_budget_bytes: 49_152,
        shared_mem_occupancy_penalty: 1.0,
    };

    let elu_spec = EluForward::<f32>::tile_spec();
    let resolve_small = |name: &str| match name {
        "BLOCK_SIZE" => block_size as i64,
        "n_elements" => small_n as i64,
        other => panic!("unexpected param {other}"),
    };
    let resolve_large = |name: &str| match name {
        "BLOCK_SIZE" => block_size as i64,
        "n_elements" => large_n as i64,
        other => panic!("unexpected param {other}"),
    };
    let small_n_ctas = small_n.div_ceil(block_size as usize) as u64;
    let large_n_ctas = large_n.div_ceil(block_size as usize) as u64;

    let predicted_small =
        cost_model.penalized_traffic(&elu_spec, size_of::<f32>(), resolve_small, small_n_ctas);
    let predicted_large =
        cost_model.penalized_traffic(&elu_spec, size_of::<f32>(), resolve_large, large_n_ctas);

    // Predicted "cost" (bytes, penalty-scaled) should rank the same way
    // measured latency does: the under-parallel small-N launch should look
    // more expensive *per byte actually moved* than the well-parallelized
    // large-N one.
    let predicted_cost_per_byte_small = predicted_small / (small_n as f64 * 4.0);
    let predicted_cost_per_byte_large = predicted_large / (large_n as f64 * 4.0);
    let measured_cost_per_byte_small = small_s / (small_n as f64 * 4.0);
    let measured_cost_per_byte_large = large_s / (large_n as f64 * 4.0);

    eprintln!(
        "predicted cost/byte: small={predicted_cost_per_byte_small:.4e} large={predicted_cost_per_byte_large:.4e} \
         | measured s/byte: small={measured_cost_per_byte_small:.4e} large={measured_cost_per_byte_large:.4e}"
    );

    assert!(
        predicted_cost_per_byte_small > predicted_cost_per_byte_large,
        "CostModel should predict the under-parallel launch as more expensive per byte"
    );
    assert!(
        measured_cost_per_byte_small > measured_cost_per_byte_large,
        "measured latency should agree: under-parallel launch is slower per byte"
    );
    Ok(())
}
