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

use super::super::Triton;
use super::super::{
    Axis, CacheModifier, DotFormat, EvictionPolicy, FpDowncastRounding, InputPrecision, MemScope,
    MemSem, PaddingOption, types as ty,
};

pub mod num;
pub mod pointer;
pub mod tensor;
pub mod types;

pub struct LlvmTriton {}

impl Triton for LlvmTriton {
    type BF16 = num::BF16;
    type BoolTensor = tensor::BoolTensor;
    type I32Tensor = tensor::I32Tensor;
    type Tensor<D: ty::Dtype> = tensor::Tensor<D>;
    type Pointer<D: ty::Dtype> = pointer::Pointer<D>;

    /*------------------------------ Programming Model ------------------------------*/

    #[inline(never)]
    fn program_id(_axis: Axis) -> i32 {
        0
    }

    #[inline(never)]
    fn num_programs(_axis: Axis) -> i32 {
        0
    }

    /*------------------------------ Creation Ops ------------------------------*/

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn arange(_start: impl Into<i32>, _end: impl Into<i32>) -> Self::I32Tensor {
        tensor::Tensor(0 as *mut i32)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn zeros<D: ty::Dtype>(_shape: &[i32]) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn zeros_like<D: ty::Dtype>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn full<D: ty::Dtype>(_shape: &[i32], _value: D) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn cast<Src: ty::Dtype, Dst: ty::Dtype>(
        _x: Self::Tensor<Src>,
        _fp_downcast_rounding: Option<FpDowncastRounding>,
        _bitcast: bool,
    ) -> Self::Tensor<Dst> {
        tensor::Tensor(0 as *mut Dst)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn cat<D: ty::Dtype>(
        _a: Self::Tensor<D>,
        _b: Self::Tensor<D>,
        _can_reorder: bool,
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    /*------------------------------ Shape Manipulation Ops ------------------------------*/

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn broadcast<D: ty::Dtype>(
        _a: Self::Tensor<D>,
        _b: Self::Tensor<D>,
    ) -> (Self::Tensor<D>, Self::Tensor<D>) {
        let t = tensor::Tensor(0 as *mut D);
        (t, t)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn broadcast_to<D: ty::Dtype>(_x: Self::Tensor<D>, _shape: &[i32]) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn expand_dims<D: ty::Dtype>(_x: Self::Tensor<D>, _axis: i32) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn permute<D: ty::Dtype>(_x: Self::Tensor<D>, _dims: &[i32]) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn reshape<D: ty::Dtype>(
        _x: Self::Tensor<D>,
        _shape: &[i32],
        _can_reorder: bool,
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn trans<D: ty::Dtype>(_x: Self::Tensor<D>, _dims: &[i32]) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn ravel<D: ty::Dtype>(_x: Self::Tensor<D>, _can_reorder: bool) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn view<D: ty::Dtype>(_x: Self::Tensor<D>, _shape: &[i32]) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn join<D: ty::Dtype>(_a: Self::Tensor<D>, _b: Self::Tensor<D>) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn interleave<D: ty::Dtype>(_a: Self::Tensor<D>, _b: Self::Tensor<D>) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn split<D: ty::Dtype>(_x: Self::Tensor<D>) -> (Self::Tensor<D>, Self::Tensor<D>) {
        let t = tensor::Tensor(0 as *mut D);
        (t, t)
    }

    /*------------------------------ Linear Algebra Ops ------------------------------*/

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn dot_scaled<D: ty::Num, S: ty::Num, O: ty::Num>(
        _lhs: Self::Tensor<D>,
        _lhs_scale: Self::Tensor<S>,
        _lhs_format: DotFormat,
        _rhs: Self::Tensor<D>,
        _rhs_scale: Self::Tensor<S>,
        _rhs_format: DotFormat,
        _acc: Option<Self::Tensor<O>>,
        _fast_math: bool,
    ) -> Self::Tensor<O> {
        tensor::Tensor(0 as *mut O)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn dot<D: ty::Num, O: ty::Num>(
        _a: Self::Tensor<D>,
        _b: Self::Tensor<D>,
        _acc: Option<Self::Tensor<O>>,
        _input_precision: Option<InputPrecision>,
        _max_num_imprecise_acc: Option<i32>,
    ) -> Self::Tensor<O> {
        tensor::Tensor(0 as *mut O)
    }

    /*------------------------------ Memory / Pointer Ops ------------------------------*/

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn make_block_ptr<D: ty::Dtype>(
        _base: Self::Pointer<D>,
        _shape: &[i32],
        _strides: &[i32],
        _offsets: &[i32],
        _block_shape: &[i32],
        _order: &[i32],
    ) -> Self::Pointer<D> {
        pointer::Pointer(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn advance<D: ty::Dtype>(_ptr: Self::Pointer<D>, _offsets: &[i32]) -> Self::Pointer<D> {
        pointer::Pointer(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn make_tensor_descriptor<D: ty::Dtype>(
        _base: Self::Pointer<D>,
        _shape: &[i32],
        _strides: &[i32],
        _block_shape: &[i32],
        _padding_option: Option<PaddingOption>,
    ) -> Self::Pointer<D> {
        pointer::Pointer(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn load_tensor_descriptor<D: ty::Dtype>(
        _desc: Self::Pointer<D>,
        _offsets: &[i32],
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    fn store_tensor_descriptor<D: ty::Dtype>(
        _desc: Self::Pointer<D>,
        _offsets: &[i32],
        _value: Self::Tensor<D>,
    ) {
        // nop
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn load<D: ty::Dtype>(
        _ptr: Self::Tensor<Self::Pointer<D>>,
        _mask: Option<Self::BoolTensor>,
        _other: Option<Self::Tensor<D>>,
        _boundary_check: &[i32],
        _padding_option: Option<PaddingOption>,
        _cache_modifier: Option<CacheModifier>,
        _eviction_policy: Option<EvictionPolicy>,
        _volatile: bool,
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    fn store<D: ty::Dtype>(
        _dest: Self::Tensor<Self::Pointer<D>>,
        _src: Self::Tensor<D>,
        _mask: Option<Self::BoolTensor>,
        _boundary_check: &[i32],
        _cache_modifier: Option<CacheModifier>,
        _eviction_policy: Option<EvictionPolicy>,
    ) {
        // nop
    }

    /*------------------------------ Indexing Ops ------------------------------*/

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn where_<D: ty::Dtype>(
        _cond: Self::BoolTensor,
        _x: Self::Tensor<D>,
        _y: Self::Tensor<D>,
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn flip<D: ty::Dtype>(_x: Self::Tensor<D>, _dim: Option<i32>) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn gather<D: ty::Dtype>(
        _src: Self::Tensor<D>,
        _index: Self::I32Tensor,
        _axis: i32,
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    /*------------------------------ Math Ops — Unary (floating-point) ------------------------------*/

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn abs<D: ty::Dtype>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn ceil<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn floor<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn cos<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn sin<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn exp<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn exp2<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn log<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn log2<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn rsqrt<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn sigmoid<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn sqrt<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn sqrt_rn<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn erf<D: ty::Float>(_x: Self::Tensor<D>) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn softmax<D: ty::Float>(
        _x: Self::Tensor<D>,
        _dim: Option<i32>,
        _keep_dims: bool,
        _ieee_rounding: bool,
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    /*------------------------------ Math Ops — Binary ------------------------------*/

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn maximum<D: ty::Num>(_x: Self::Tensor<D>, _y: Self::Tensor<D>) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn minimum<D: ty::Num>(_x: Self::Tensor<D>, _y: Self::Tensor<D>) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn clamp<D: ty::Num>(
        _x: Self::Tensor<D>,
        _lo: Self::Tensor<D>,
        _hi: Self::Tensor<D>,
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn fma<D: ty::Float>(
        _x: Self::Tensor<D>,
        _y: Self::Tensor<D>,
        _z: Self::Tensor<D>,
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn fdiv<D: ty::Float>(
        _x: Self::Tensor<D>,
        _y: Self::Tensor<D>,
        _ieee_rounding: bool,
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn div_rn<D: ty::Float>(_x: Self::Tensor<D>, _y: Self::Tensor<D>) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn umulhi(_x: Self::Tensor<u32>, _y: Self::Tensor<u32>) -> Self::Tensor<u32> {
        tensor::Tensor(0 as *mut u32)
    }

    #[inline(never)]
    fn cdiv(_x: i32, _div: i32) -> i32 {
        0
    }

    #[inline(never)]
    fn swizzle2d(_i: i32, _j: i32, _size_i: i32, _size_j: i32, _size_g: i32) -> (i32, i32) {
        (0, 0)
    }

    /*------------------------------ Reduction Ops ------------------------------*/

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn sum<D: ty::Num>(
        _x: Self::Tensor<D>,
        _axis: Option<i32>,
        _keep_dims: bool,
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn max<D: ty::Num>(
        _x: Self::Tensor<D>,
        _axis: Option<i32>,
        _keep_dims: bool,
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn max_with_indices<D: ty::Num>(
        _x: Self::Tensor<D>,
        _axis: i32,
        _tie_break_left: bool,
        _keep_dims: bool,
    ) -> (Self::Tensor<D>, Self::I32Tensor) {
        (tensor::Tensor(0 as *mut D), tensor::Tensor(0 as *mut i32))
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn min<D: ty::Num>(
        _x: Self::Tensor<D>,
        _axis: Option<i32>,
        _keep_dims: bool,
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn min_with_indices<D: ty::Num>(
        _x: Self::Tensor<D>,
        _axis: i32,
        _tie_break_left: bool,
        _keep_dims: bool,
    ) -> (Self::Tensor<D>, Self::I32Tensor) {
        (tensor::Tensor(0 as *mut D), tensor::Tensor(0 as *mut i32))
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn argmax<D: ty::Num>(
        _x: Self::Tensor<D>,
        _axis: i32,
        _tie_break_left: bool,
        _keep_dims: bool,
    ) -> Self::I32Tensor {
        tensor::Tensor(0 as *mut i32)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn argmin<D: ty::Num>(
        _x: Self::Tensor<D>,
        _axis: i32,
        _tie_break_left: bool,
        _keep_dims: bool,
    ) -> Self::I32Tensor {
        tensor::Tensor(0 as *mut i32)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn xor_sum<D: ty::Int>(
        _x: Self::Tensor<D>,
        _axis: Option<i32>,
        _keep_dims: bool,
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    /*------------------------------ Scan / Sort Ops ------------------------------*/

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn cumsum<D: ty::Num>(_x: Self::Tensor<D>, _axis: i32, _reverse: bool) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn cumprod<D: ty::Num>(_x: Self::Tensor<D>, _axis: i32, _reverse: bool) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn sort<D: ty::Num>(
        _x: Self::Tensor<D>,
        _dim: Option<i32>,
        _descending: bool,
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn histogram(
        _x: Self::I32Tensor,
        _num_bins: i32,
        _mask: Option<Self::BoolTensor>,
    ) -> Self::I32Tensor {
        tensor::Tensor(0 as *mut i32)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn reduce<D: ty::Dtype, O: ty::Dtype>(
        _x: Self::Tensor<D>,
        _axis: i32,
        _combine_fn: fn(Self::Tensor<O>, Self::Tensor<O>) -> Self::Tensor<O>,
        _keep_dims: bool,
    ) -> Self::Tensor<O> {
        tensor::Tensor(0 as *mut O)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn associative_scan<D: ty::Dtype>(
        _x: Self::Tensor<D>,
        _axis: i32,
        _combine_fn: fn(Self::Tensor<D>, Self::Tensor<D>) -> Self::Tensor<D>,
        _reverse: bool,
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    /*------------------------------ Atomic Ops ------------------------------*/

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn atomic_add<D: ty::Num>(
        _ptr: Self::Tensor<Self::Pointer<D>>,
        _val: Self::Tensor<D>,
        _mask: Option<Self::BoolTensor>,
        _sem: Option<MemSem>,
        _scope: Option<MemScope>,
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn atomic_and<D: ty::Int>(
        _ptr: Self::Tensor<Self::Pointer<D>>,
        _val: Self::Tensor<D>,
        _mask: Option<Self::BoolTensor>,
        _sem: Option<MemSem>,
        _scope: Option<MemScope>,
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn atomic_or<D: ty::Int>(
        _ptr: Self::Tensor<Self::Pointer<D>>,
        _val: Self::Tensor<D>,
        _mask: Option<Self::BoolTensor>,
        _sem: Option<MemSem>,
        _scope: Option<MemScope>,
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn atomic_xor<D: ty::Int>(
        _ptr: Self::Tensor<Self::Pointer<D>>,
        _val: Self::Tensor<D>,
        _mask: Option<Self::BoolTensor>,
        _sem: Option<MemSem>,
        _scope: Option<MemScope>,
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn atomic_max<D: ty::Num>(
        _ptr: Self::Tensor<Self::Pointer<D>>,
        _val: Self::Tensor<D>,
        _mask: Option<Self::BoolTensor>,
        _sem: Option<MemSem>,
        _scope: Option<MemScope>,
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn atomic_min<D: ty::Num>(
        _ptr: Self::Tensor<Self::Pointer<D>>,
        _val: Self::Tensor<D>,
        _mask: Option<Self::BoolTensor>,
        _sem: Option<MemSem>,
        _scope: Option<MemScope>,
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn atomic_xchg<D: ty::Dtype>(
        _ptr: Self::Tensor<Self::Pointer<D>>,
        _val: Self::Tensor<D>,
        _mask: Option<Self::BoolTensor>,
        _sem: Option<MemSem>,
        _scope: Option<MemScope>,
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn atomic_cas<D: ty::Dtype>(
        _ptr: Self::Tensor<Self::Pointer<D>>,
        _cmp: Self::Tensor<D>,
        _val: Self::Tensor<D>,
        _sem: Option<MemSem>,
        _scope: Option<MemScope>,
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    /*------------------------------ Random Number Generation ------------------------------*/

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn rand(_seed: u32, _offsets: Self::I32Tensor, _n_rounds: i32) -> Self::Tensor<f32> {
        tensor::Tensor(0 as *mut f32)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn randn(_seed: u32, _offsets: Self::I32Tensor, _n_rounds: i32) -> Self::Tensor<f32> {
        tensor::Tensor(0 as *mut f32)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn randint(_seed: u32, _offsets: Self::I32Tensor, _n_rounds: i32) -> Self::I32Tensor {
        tensor::Tensor(0 as *mut i32)
    }

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn randint4x(
        _seed: u32,
        _offsets: Self::I32Tensor,
        _n_rounds: i32,
    ) -> (
        Self::I32Tensor,
        Self::I32Tensor,
        Self::I32Tensor,
        Self::I32Tensor,
    ) {
        let t = tensor::Tensor(0 as *mut i32);
        (t, t, t, t)
    }

    /*------------------------------ Compiler Hint Ops ------------------------------*/

    #[inline(always)]
    fn multiple_of<D: ty::Dtype>(x: Self::Tensor<D>, _values: &[i32]) -> Self::Tensor<D> {
        x
    }

    #[inline(always)]
    fn max_contiguous<D: ty::Dtype>(x: Self::Tensor<D>, _values: &[i32]) -> Self::Tensor<D> {
        x
    }

    #[inline(always)]
    fn max_constancy<D: ty::Dtype>(x: Self::Tensor<D>, _values: &[i32]) -> Self::Tensor<D> {
        x
    }

    /*------------------------------ Inline Assembly ------------------------------*/

    #[inline(never)]
    #[allow(clippy::zero_ptr)]
    fn inline_asm_elementwise<D: ty::Dtype>(
        _asm: &str,
        _constraints: &str,
        _is_pure: bool,
        _pack: i32,
    ) -> Self::Tensor<D> {
        tensor::Tensor(0 as *mut D)
    }

    /*------------------------------ Compiler Hint Ops ------------------------------*/

    #[inline(always)]
    fn assume(_cond: Self::BoolTensor) {
        // hint only — no-op in dummy implementation
    }

    /*------------------------------ Debug Ops ------------------------------*/

    #[inline(always)]
    fn debug_barrier() {
        // no-op in dummy implementation
    }

    #[inline(always)]
    fn device_assert(_cond: Self::BoolTensor, _msg: &str, _mask: Option<Self::BoolTensor>) {
        // no-op in dummy implementation
    }

    #[inline(always)]
    fn device_print<D: ty::Dtype>(_prefix: &str, _val: Self::Tensor<D>, _hex: bool) {
        // no-op in dummy implementation
    }

    #[inline(always)]
    fn static_assert(_cond: bool, _msg: &str) {
        // no-op in dummy implementation (compiler lowering handles this)
    }

    #[inline(always)]
    fn static_print(_msg: &str) {
        // no-op in dummy implementation (compiler lowering handles this)
    }
}
