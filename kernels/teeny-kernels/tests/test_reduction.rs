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
use insta::assert_debug_snapshot;
use teeny_compiler::compiler::{driver::cuda::compile_kernel, target::cuda::Target};
use teeny_core::device::program::Kernel;

#[cfg(feature = "cuda")]
use teeny_cuda::{compiler::target::Capability, errors::Result, testing};

const TOL: f32 = 1e-4;

// ── Source snapshot tests ─────────────────────────────────────────────────────

#[test]
fn test_reduce_sum_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::tensor::reduction::ReduceSumForward::<f32>::new(1024);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm89);
    compile_kernel(&kernel, &target, true)?;
    assert_debug_snapshot!("reduce_sum_forward_source", kernel.source());
    Ok(())
}

#[test]
fn test_reduce_max_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::tensor::reduction::ReduceMaxForward::<f32>::new(1024);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm89);
    compile_kernel(&kernel, &target, true)?;
    assert_debug_snapshot!("reduce_max_forward_source", kernel.source());
    Ok(())
}

#[test]
fn test_cum_sum_source() -> anyhow::Result<()> {
    dotenv().ok();
    let kernel = teeny_kernels::nn::tensor::reduction::CumSumForward::<f32>::new(1024);
    let target = Target::new(teeny_cuda::compiler::target::Capability::Sm89);
    compile_kernel(&kernel, &target, true)?;
    assert_debug_snapshot!("cum_sum_forward_source", kernel.source());
    Ok(())
}
