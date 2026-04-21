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

use std::path::PathBuf;

use insta::assert_debug_snapshot;
use teeny_compiler::compiler::{backend::llvm::compiler::LlvmCompiler, target::cuda::Target};
use teeny_core::{
    compiler::Compiler,
    context::program::Kernel,
    dtype::{Dtype, Float},
};
use teeny_macros::kernel;
use teeny_triton::triton::{
    Axis, CacheModifier, DotFormat, EvictionPolicy, FpDowncastRounding, InputPrecision, MemScope,
    MemSem, PaddingOption, Triton,
};

use teeny_cuda::compiler::target::Capability;

#[kernel]
#[allow(unused)]
fn kitchen_sink<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    output_ptr: T::Pointer<D>,
    n_elements: i32,
) {
    fn combine_num<TT: Triton, DD: Dtype>(
        lhs: TT::Tensor<DD>,
        rhs: TT::Tensor<DD>,
    ) -> TT::Tensor<DD> {
        lhs + rhs
    }

    #[allow(clippy::empty_loop)]
    fn dummy_bool_tensor<TT: Triton>() -> TT::BoolTensor {
        loop {} // dummy bool should never be evaluated
    }

    #[allow(clippy::empty_loop)]
    fn dummy_value<DD: Dtype>() -> DD {
        loop {} // dummy value should never be evaluated
    }

    let _pid_x = T::program_id(Axis::X);
    let _pid_y = T::program_id(Axis::Y);
    let _pid_z = T::program_id(Axis::Z);
    let _nprog_x = T::num_programs(Axis::X);

    let r = T::arange(0, BLOCK_SIZE);
    let _ = r + n_elements;
    let _ = r - 1;
    let _ = r * 2;

    let z = T::zeros::<D>(&[BLOCK_SIZE]);
    if false {
        let _ = T::full::<D>(&[BLOCK_SIZE], dummy_value::<D>());
    }
    let zl = T::zeros_like(z);
    let _ = T::full::<D>(&[BLOCK_SIZE], dummy_value::<D>());
    let casted = T::cast::<D, D>(z, Some(FpDowncastRounding::Rtne), false);
    let _casted_rtz = T::cast::<D, D>(casted, Some(FpDowncastRounding::Rtz), true);
    let cat = T::cat(z, zl, true);

    let (ba, bb) = T::broadcast(cat, cat);
    let bto = T::broadcast_to(ba, &[BLOCK_SIZE]);
    let ex = T::expand_dims(bto, 0);
    let p = T::permute(ex, &[0]);
    let rs = T::reshape(p, &[BLOCK_SIZE], false);
    let tr = T::trans(rs, &[0]);
    let rv = T::ravel(tr, false);
    let vw = T::view(rv, &[BLOCK_SIZE]);
    let jn = T::join(vw, bb);
    let il = T::interleave(jn, jn);
    let (sp0, sp1) = T::split(il);

    let dot = T::dot::<D, D>(sp0, sp1, None, Some(InputPrecision::TF32), Some(1));
    let _dot_tf32x3 = T::dot::<D, D>(dot, dot, None, Some(InputPrecision::TF32x3), None);
    let _dot_ieee = T::dot::<D, D>(dot, dot, None, Some(InputPrecision::IEEE), None);
    let scale = zl;
    let _dot_scaled = T::dot_scaled::<D, D, D>(
        dot,
        scale,
        DotFormat::E4M3,
        dot,
        scale,
        DotFormat::E5M2,
        None,
        true,
    );
    let _ = T::dot_scaled::<D, D, D>(
        dot,
        scale,
        DotFormat::E2M1x2,
        dot,
        scale,
        DotFormat::E2M1x4,
        None,
        false,
    );
    let _ = T::dot_scaled::<D, D, D>(
        dot,
        scale,
        DotFormat::BF16x2,
        dot,
        scale,
        DotFormat::Int8,
        None,
        false,
    );
    let _ = T::dot_scaled::<D, D, D>(
        dot,
        scale,
        DotFormat::UInt8,
        dot,
        scale,
        DotFormat::E4M3,
        None,
        false,
    );

    let block_ptr = T::make_block_ptr(x_ptr, &[BLOCK_SIZE], &[1], &[0], &[BLOCK_SIZE], &[0]);
    let block_ptr2 = T::advance(block_ptr, &[1]);
    let tdesc = T::make_tensor_descriptor(
        y_ptr,
        &[BLOCK_SIZE],
        &[1],
        &[BLOCK_SIZE],
        Some(PaddingOption::Zero),
    );
    let tdesc_nan = T::make_tensor_descriptor(
        y_ptr,
        &[BLOCK_SIZE],
        &[1],
        &[BLOCK_SIZE],
        Some(PaddingOption::Nan),
    );
    let tdv = T::load_tensor_descriptor(tdesc, &[0]);
    T::store_tensor_descriptor(tdesc_nan, &[0], tdv);

    let ptrs = T::zeros::<T::Pointer<D>>(&[BLOCK_SIZE]);
    let loaded = T::load::<D, 1>(
        ptrs,
        None,
        Some(T::zeros::<D>(&[BLOCK_SIZE])),
        &[0],
        Some(PaddingOption::Zero),
        Some(CacheModifier::Ca),
        Some(EvictionPolicy::EvictFirst),
        false,
    );
    let _ = T::load::<D, 1>(
        ptrs,
        None,
        None,
        &[0],
        Some(PaddingOption::Nan),
        Some(CacheModifier::Cg),
        Some(EvictionPolicy::EvictLast),
        true,
    );
    let _ = T::load::<D, 1>(
        ptrs,
        None,
        None,
        &[0],
        None,
        Some(CacheModifier::Cv),
        Some(EvictionPolicy::NoEvict),
        false,
    );
    T::store::<D, 1>(
        ptrs,
        loaded,
        None,
        &[0],
        Some(CacheModifier::Wb),
        Some(EvictionPolicy::NoEvict),
    );
    T::store::<D, 1>(ptrs, loaded, None, &[0], Some(CacheModifier::Cs), None);

    if false {
        let cond: T::BoolTensor = dummy_bool_tensor::<T>();
        let _ = T::where_(cond, loaded, loaded);
        T::assume(cond);
        T::device_assert(cond, "kitchen_sink", Some(cond));
    }

    let fl = T::flip(loaded, Some(0));
    let _ = T::gather(fl, r, 0);

    let _abs = T::abs(loaded);
    let ceil = T::ceil(loaded);
    let floor = T::floor(ceil);
    let cos = T::cos(floor);
    let sin = T::sin(cos);
    let exp = T::exp(sin);
    let exp2 = T::exp2(exp);
    let log = T::log(exp2);
    let log2 = T::log2(log);
    let rsqrt = T::rsqrt(log2);
    let sig = T::sigmoid(rsqrt);
    let sqrt = T::sqrt(sig);
    let sqrt_rn = T::sqrt_rn(sqrt);
    let erf = T::erf(sqrt_rn);
    let smax = T::softmax(erf, Some(0), true, true);

    let mx = T::maximum(smax, smax);
    let mn = T::minimum(mx, smax);
    let cl = T::clamp(mn, smax, mx);
    let fm = T::fma(cl, smax, mx);
    let fd = T::fdiv(fm, smax, true);
    let dr = T::div_rn(fd, smax);
    let _cd = T::cdiv(n_elements, BLOCK_SIZE);
    let _swz = T::swizzle2d(0, 0, BLOCK_SIZE, BLOCK_SIZE, 1);

    let _sum = T::sum(dr, Some(0), true);
    let _max = T::max(dr, None, false);
    let (_maxv, _maxi) = T::max_with_indices(dr, 0, true, false);
    let _min = T::min(dr, Some(0), false);
    let (_minv, _mini) = T::min_with_indices(dr, 0, true, false);
    let _argmax = T::argmax(dr, 0, true, false);
    let _argmin = T::argmin(dr, 0, true, false);
    let _xors = T::xor_sum(T::zeros::<i32>(&[BLOCK_SIZE]), Some(0), false);

    let _cumsum = T::cumsum(dr, 0, false);
    let _cumprod = T::cumprod(dr, 0, true);
    let _sort = T::sort(dr, Some(0), true);
    let _hist = T::histogram(r, BLOCK_SIZE, None);
    let _reduced = T::reduce::<D, D>(dr, 0, combine_num::<T, D>, false);
    let _scan = T::associative_scan::<D>(dr, 0, combine_num::<T, D>, true);

    let aptrs = T::zeros::<T::Pointer<D>>(&[BLOCK_SIZE]);
    let _ = T::atomic_add(aptrs, dr, None, Some(MemSem::Relaxed), Some(MemScope::Cta));
    let _ = T::atomic_max(aptrs, dr, None, Some(MemSem::Acquire), Some(MemScope::Gpu));
    let _ = T::atomic_min(aptrs, dr, None, Some(MemSem::Release), Some(MemScope::Sys));
    let _ = T::atomic_xchg(aptrs, dr, None, Some(MemSem::AcqRel), Some(MemScope::Gpu));
    let _ = T::atomic_cas(aptrs, dr, dr, Some(MemSem::AcqRel), Some(MemScope::Gpu));

    let iptrs = T::zeros::<T::Pointer<i32>>(&[BLOCK_SIZE]);
    let ival = T::zeros::<i32>(&[BLOCK_SIZE]);
    let _ = T::atomic_and(
        iptrs,
        ival,
        None,
        Some(MemSem::Relaxed),
        Some(MemScope::Cta),
    );
    let _ = T::atomic_or(
        iptrs,
        ival,
        None,
        Some(MemSem::Acquire),
        Some(MemScope::Gpu),
    );
    let _ = T::atomic_xor(
        iptrs,
        ival,
        None,
        Some(MemSem::Release),
        Some(MemScope::Sys),
    );

    let u32a = T::zeros::<u32>(&[BLOCK_SIZE]);
    let u32b = T::zeros::<u32>(&[BLOCK_SIZE]);
    let _umulhi = T::umulhi(u32a, u32b);

    let _rand = T::rand(123, r, 10);
    let _randn = T::randn(123, r, 10);
    let _randi = T::randint(123, r, 10);
    let _rand4 = T::randint4x(123, r, 10);

    let _asm = T::inline_asm_elementwise::<D>("", "", true, 1);

    let mo = T::multiple_of(dr, &[1]);
    let mc = T::max_contiguous(mo, &[1]);
    let mconst = T::max_constancy(mc, &[1]);

    T::debug_barrier();
    T::device_print("val=", mconst, false);
    T::static_assert(true, "BLOCK_SIZE must be positive");
    T::static_print("kitchen_sink");

    let out_ptrs = T::zeros::<T::Pointer<D>>(&[BLOCK_SIZE]);
    let _ = T::make_block_ptr(output_ptr, &[BLOCK_SIZE], &[1], &[0], &[BLOCK_SIZE], &[0]);
    T::store::<D, 1>(out_ptrs, mconst, None, &[0], None, None);
    let _ = block_ptr2;
}

#[test]
fn test_kitchen_sink() -> anyhow::Result<()> {
    let kernel = KitchenSink::<f32, 1024>::new();
    let rustc_path = std::env::var("TEENY_RUSTC_PATH").expect("TEENY_RUSTC_PATH must be set");
    let cache_dir =
        std::env::var("TEENY_CACHE_DIR").unwrap_or_else(|_| "/tmp/teenygrad_rustc".to_string());
    let compiler = LlvmCompiler::new(rustc_path, cache_dir)?;
    let target = Target::new(Capability::Sm90);
    let ptx_path: PathBuf = compiler.compile(&kernel, &target, true)?.into();
    let mlir = std::fs::read_to_string(ptx_path.with_extension("mlir"))?;

    assert_debug_snapshot!("kitchen_sink_source", kernel.source());
    assert_debug_snapshot!("kitchen_sink_mlir", mlir.trim());

    Ok(())
}
