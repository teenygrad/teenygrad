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

use core::ops::{Add, Div, Mul, Neg, Sub};
pub use core::ops::{BitAnd, BitOr};

use self::types::{self as ty};

pub mod llvm;
pub mod types;

pub use types::*;

/*------------------------------ Parameter Enums ------------------------------*/

#[repr(i32)]
pub enum Axis {
    X = 0,
    Y = 1,
    Z = 2,
}

/// Padding value applied to out-of-bounds lanes when using `boundary_check` in `load`.
pub enum PaddingOption {
    /// Pad with zero.
    Zero,
    /// Pad with NaN.
    Nan,
}

/// L1/L2 cache behaviour for load and store instructions.
pub enum CacheModifier {
    /// Cache at all levels (L1 + L2).
    Ca,
    /// Cache at global level only (L2, bypass L1).
    Cg,
    /// Volatile — don't cache, always fetch from memory.
    Cv,
    /// Write-back at all coherent levels.
    Wb,
    /// Streaming — likely accessed once.
    Cs,
}

/// Cache eviction priority hint for load and store instructions.
pub enum EvictionPolicy {
    EvictFirst,
    EvictLast,
    NoEvict,
}

/// Tensor-core precision mode for `dot` on `f32 × f32` inputs.
pub enum InputPrecision {
    /// TF32 precision (default on devices with Tensor Cores).
    TF32,
    /// Emulate higher precision using three TF32 dot products.
    TF32x3,
    /// Full IEEE-754 precision.
    IEEE,
}

/// Rounding mode used when down-casting floating-point types in `cast`.
pub enum FpDowncastRounding {
    /// Round to nearest, ties to even.
    Rtne,
    /// Round towards zero (truncate).
    Rtz,
}

/// Input format for scaled dot-product (`dot_scaled`).
pub enum DotFormat {
    E4M3,
    E5M2,
    E2M1x2,
    E2M1x4,
    BF16x2,
    Int8,
    UInt8,
}

/// Memory ordering semantics for atomic operations.
pub enum MemSem {
    Relaxed,
    Acquire,
    Release,
    /// Acquire + Release (default).
    AcqRel,
}

/// Synchronization scope for atomic operations.
pub enum MemScope {
    /// Cooperative thread array (thread block).
    Cta,
    /// All threads on the GPU (default).
    Gpu,
    /// All threads in the system.
    Sys,
}

/*------------------------------ Triton Trait ------------------------------*/

pub trait Triton
where
    Self::I32Tensor: Add<i32, Output = Self::I32Tensor>,
    Self::I32Tensor: Sub<i32, Output = Self::I32Tensor>,
    Self::I32Tensor: Mul<i32, Output = Self::I32Tensor>,
    Self::BoolTensor: BitAnd<Output = Self::BoolTensor>,
    Self::BoolTensor: BitOr<Output = Self::BoolTensor>,
{
    type BF16: ty::BF16;
    type BoolTensor: Copy + Clone;
    type I32Tensor: Copy + Clone;
    type Tensor<D: ty::Dtype>: Copy + Clone
        + Add<Self::Tensor<D>, Output = Self::Tensor<D>>
        + Sub<Self::Tensor<D>, Output = Self::Tensor<D>>
        + Mul<Self::Tensor<D>, Output = Self::Tensor<D>>
        + Div<Self::Tensor<D>, Output = Self::Tensor<D>>
        + Neg<Output = Self::Tensor<D>>;
    type Pointer<D: ty::Dtype>: Copy + Clone + ty::Dtype + Add<Self::Pointer<D>, Output = Self::Pointer<D>>;

    /*------------------------------ Programming Model ------------------------------*/

    fn program_id(axis: Axis) -> i32;

    fn num_programs(axis: Axis) -> i32;

    /*------------------------------ Creation Ops ------------------------------*/

    fn arange(start: impl Into<i32>, end: impl Into<i32>) -> Self::I32Tensor;

    fn zeros<D: ty::Dtype>(shape: &[i32]) -> Self::Tensor<D>;

    fn zeros_like<D: ty::Dtype>(x: Self::Tensor<D>) -> Self::Tensor<D>;

    fn full<D: ty::Dtype>(shape: &[i32], value: D) -> Self::Tensor<D>;

    /// Cast a tensor to a different dtype.
    ///
    /// - `fp_downcast_rounding`: rounding mode when narrowing float types (default `None` = unspecified).
    /// - `bitcast`: reinterpret bits without conversion (default `false`).
    fn cast<Src: ty::Dtype, Dst: ty::Dtype>(
        x: Self::Tensor<Src>,
        fp_downcast_rounding: Option<FpDowncastRounding>,
        bitcast: bool,
    ) -> Self::Tensor<Dst>;

    /// Concatenate two tensors.
    ///
    /// - `can_reorder`: allow the compiler to reorder elements (default `false`).
    fn cat<D: ty::Dtype>(
        a: Self::Tensor<D>,
        b: Self::Tensor<D>,
        can_reorder: bool,
    ) -> Self::Tensor<D>;

    /*------------------------------ Shape Manipulation Ops ------------------------------*/

    /// Broadcast two tensors to a common compatible shape.
    fn broadcast<D: ty::Dtype>(
        a: Self::Tensor<D>,
        b: Self::Tensor<D>,
    ) -> (Self::Tensor<D>, Self::Tensor<D>);

    fn broadcast_to<D: ty::Dtype>(x: Self::Tensor<D>, shape: &[i32]) -> Self::Tensor<D>;

    fn expand_dims<D: ty::Dtype>(x: Self::Tensor<D>, axis: i32) -> Self::Tensor<D>;

    fn permute<D: ty::Dtype>(x: Self::Tensor<D>, dims: &[i32]) -> Self::Tensor<D>;

    /// Reshape a tensor.
    ///
    /// - `can_reorder`: allow element reordering during reshape (default `false`).
    fn reshape<D: ty::Dtype>(
        x: Self::Tensor<D>,
        shape: &[i32],
        can_reorder: bool,
    ) -> Self::Tensor<D>;

    /// Permute dimensions. Alias for `permute`.
    fn trans<D: ty::Dtype>(x: Self::Tensor<D>, dims: &[i32]) -> Self::Tensor<D>;

    /// Flatten to 1-D.
    ///
    /// - `can_reorder`: allow element reordering (default `false`).
    fn ravel<D: ty::Dtype>(x: Self::Tensor<D>, can_reorder: bool) -> Self::Tensor<D>;

    /// View with a new shape (order not preserved).
    fn view<D: ty::Dtype>(x: Self::Tensor<D>, shape: &[i32]) -> Self::Tensor<D>;

    /// Join two tensors along a new minor dimension.
    fn join<D: ty::Dtype>(a: Self::Tensor<D>, b: Self::Tensor<D>) -> Self::Tensor<D>;

    /// Interleave two tensors along their last dimension.
    fn interleave<D: ty::Dtype>(a: Self::Tensor<D>, b: Self::Tensor<D>) -> Self::Tensor<D>;

    /// Split a tensor in two along its last dimension (which must have size 2).
    fn split<D: ty::Dtype>(x: Self::Tensor<D>) -> (Self::Tensor<D>, Self::Tensor<D>);

    /*------------------------------ Linear Algebra Ops ------------------------------*/

    /// Matrix (or batched matrix) multiply.
    ///
    /// - `acc`: optional accumulator tensor added to the result.
    /// - `input_precision`: Tensor Core precision for `f32 × f32` (default `None` = TF32 on capable hardware).
    /// - `max_num_imprecise_acc`: limit on imprecise accumulations (default `None`).
    fn dot<D: ty::Num, O: ty::Num>(
        a: Self::Tensor<D>,
        b: Self::Tensor<D>,
        acc: Option<Self::Tensor<O>>,
        input_precision: Option<InputPrecision>,
        max_num_imprecise_acc: Option<i32>,
    ) -> Self::Tensor<O>;

    /// Scaled mixed-precision matrix multiply (FP8 / narrow formats).
    ///
    /// - `acc`: optional accumulator (default `None`).
    /// - `fast_math`: allow reduced precision accumulation (default `false`).
    fn dot_scaled<D: ty::Num, S: ty::Num, O: ty::Num>(
        lhs: Self::Tensor<D>,
        lhs_scale: Self::Tensor<S>,
        lhs_format: DotFormat,
        rhs: Self::Tensor<D>,
        rhs_scale: Self::Tensor<S>,
        rhs_format: DotFormat,
        acc: Option<Self::Tensor<O>>,
        fast_math: bool,
    ) -> Self::Tensor<O>;

    /*------------------------------ Memory / Pointer Ops ------------------------------*/

    /// Create a block pointer encoding shape, strides, offsets, and tile shape.
    fn make_block_ptr<D: ty::Dtype>(
        base: Self::Pointer<D>,
        shape: &[i32],
        strides: &[i32],
        offsets: &[i32],
        block_shape: &[i32],
        order: &[i32],
    ) -> Self::Pointer<D>;

    /// Advance a block pointer by the given per-dimension offsets.
    fn advance<D: ty::Dtype>(ptr: Self::Pointer<D>, offsets: &[i32]) -> Self::Pointer<D>;

    /// Create a tensor descriptor for TMA (Tensor Memory Accelerator) operations.
    ///
    /// - `padding_option`: out-of-bounds padding behaviour (default `PaddingOption::Zero`).
    fn make_tensor_descriptor<D: ty::Dtype>(
        base: Self::Pointer<D>,
        shape: &[i32],
        strides: &[i32],
        block_shape: &[i32],
        padding_option: Option<PaddingOption>,
    ) -> Self::Pointer<D>;

    /// Load a tile from memory using a tensor descriptor and per-dimension offsets.
    fn load_tensor_descriptor<D: ty::Dtype>(
        desc: Self::Pointer<D>,
        offsets: &[i32],
    ) -> Self::Tensor<D>;

    /// Store a tile to memory using a tensor descriptor and per-dimension offsets.
    fn store_tensor_descriptor<D: ty::Dtype>(
        desc: Self::Pointer<D>,
        offsets: &[i32],
        value: Self::Tensor<D>,
    );

    /// Load a tensor from memory.
    ///
    /// - `mask`: when `Some`, lanes where mask is `false` are not loaded (default `None` = unconditional).
    /// - `other`: fill value for masked-off lanes (default `None` = undefined).
    /// - `boundary_check`: dimensions to check for out-of-bounds (block-pointer mode only, default `&[]`).
    /// - `padding_option`: fill for out-of-bounds lanes in block-pointer mode (default `None`).
    /// - `cache_modifier`: L1/L2 cache behaviour (default `None`).
    /// - `eviction_policy`: eviction priority hint (default `None`).
    /// - `volatile`: always fetch fresh from memory (default `false`).
    fn load<D: ty::Dtype, const N: usize>(
        ptr: Self::Tensor<Self::Pointer<D>>,
        mask: Option<Self::BoolTensor>,
        other: Option<Self::Tensor<D>>,
        boundary_check: &[i32; N],
        padding_option: Option<PaddingOption>,
        cache_modifier: Option<CacheModifier>,
        eviction_policy: Option<EvictionPolicy>,
        volatile: bool,
    ) -> Self::Tensor<D>;

    /// Store a tensor to memory.
    ///
    /// - `mask`: when `Some`, lanes where mask is `false` are not stored (default `None` = unconditional).
    /// - `boundary_check`: dimensions to check for out-of-bounds (block-pointer mode only, default `&[]`).
    /// - `cache_modifier`: L1/L2 cache behaviour (default `None`).
    /// - `eviction_policy`: eviction priority hint (default `None`).
    fn store<D: ty::Dtype, const N: usize>(
        dest: Self::Tensor<Self::Pointer<D>>,
        src: Self::Tensor<D>,
        mask: Option<Self::BoolTensor>,
        boundary_check: &[i32; N],
        cache_modifier: Option<CacheModifier>,
        eviction_policy: Option<EvictionPolicy>,
    );

    /*------------------------------ Comparison Ops ------------------------------*/

    /// Element-wise less-than between two tensors.
    fn lt<D: ty::Num>(x: Self::Tensor<D>, y: Self::Tensor<D>) -> Self::BoolTensor;
    /// Element-wise less-than-or-equal between two tensors.
    fn le<D: ty::Num>(x: Self::Tensor<D>, y: Self::Tensor<D>) -> Self::BoolTensor;
    /// Element-wise greater-than between two tensors.
    fn gt<D: ty::Num>(x: Self::Tensor<D>, y: Self::Tensor<D>) -> Self::BoolTensor;
    /// Element-wise greater-than-or-equal between two tensors.
    fn ge<D: ty::Num>(x: Self::Tensor<D>, y: Self::Tensor<D>) -> Self::BoolTensor;
    /// Element-wise equality between two tensors.
    fn eq<D: ty::Num>(x: Self::Tensor<D>, y: Self::Tensor<D>) -> Self::BoolTensor;
    /// Element-wise inequality between two tensors.
    fn ne<D: ty::Num>(x: Self::Tensor<D>, y: Self::Tensor<D>) -> Self::BoolTensor;

    /// Element-wise less-than against a scalar.
    fn lt_scalar<D: ty::Num>(x: Self::Tensor<D>, y: D) -> Self::BoolTensor;
    /// Element-wise less-than-or-equal against a scalar.
    fn le_scalar<D: ty::Num>(x: Self::Tensor<D>, y: D) -> Self::BoolTensor;
    /// Element-wise greater-than against a scalar.
    fn gt_scalar<D: ty::Num>(x: Self::Tensor<D>, y: D) -> Self::BoolTensor;
    /// Element-wise greater-than-or-equal against a scalar.
    fn ge_scalar<D: ty::Num>(x: Self::Tensor<D>, y: D) -> Self::BoolTensor;
    /// Element-wise equality against a scalar.
    fn eq_scalar<D: ty::Num>(x: Self::Tensor<D>, y: D) -> Self::BoolTensor;
    /// Element-wise inequality against a scalar.
    fn ne_scalar<D: ty::Num>(x: Self::Tensor<D>, y: D) -> Self::BoolTensor;

    /*------------------------------ Indexing Ops ------------------------------*/

    /// Conditional element selection — corresponds to `tl.where`.
    /// Named `where_` to avoid collision with the Rust keyword `where`.
    fn where_<D: ty::Dtype>(
        cond: Self::BoolTensor,
        x: Self::Tensor<D>,
        y: Self::Tensor<D>,
    ) -> Self::Tensor<D>;

    /// Reverse a tensor along `dim`. `None` reverses all dimensions.
    fn flip<D: ty::Dtype>(x: Self::Tensor<D>, dim: Option<i32>) -> Self::Tensor<D>;

    fn gather<D: ty::Dtype>(
        src: Self::Tensor<D>,
        index: Self::I32Tensor,
        axis: i32,
    ) -> Self::Tensor<D>;

    /*------------------------------ Math Ops — Unary ------------------------------*/

    /// Element-wise absolute value.
    fn abs<D: ty::Dtype>(x: Self::Tensor<D>) -> Self::Tensor<D>;
    fn ceil<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;
    fn floor<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;
    fn cos<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;
    fn sin<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;
    fn exp<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;
    fn exp2<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;
    fn log<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;
    fn log2<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;
    fn rsqrt<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;
    fn sigmoid<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;
    fn sqrt<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;
    fn sqrt_rn<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;
    fn erf<D: ty::Float>(x: Self::Tensor<D>) -> Self::Tensor<D>;

    /*------------------------------ Math Ops — Float (higher-level) ------------------------------*/

    /// Numerically-stable softmax along `dim`. `dim = None` defaults to the last dimension.
    ///
    /// - `keep_dims`: retain the reduced dimension with length 1 (default `false`).
    /// - `ieee_rounding`: use IEEE-754 rounding (default `false`).
    fn softmax<D: ty::Float>(
        x: Self::Tensor<D>,
        dim: Option<i32>,
        keep_dims: bool,
        ieee_rounding: bool,
    ) -> Self::Tensor<D>;

    /*------------------------------ Math Ops — Binary ------------------------------*/

    fn maximum<D: ty::Num>(x: Self::Tensor<D>, y: Self::Tensor<D>) -> Self::Tensor<D>;
    fn minimum<D: ty::Num>(x: Self::Tensor<D>, y: Self::Tensor<D>) -> Self::Tensor<D>;

    fn clamp<D: ty::Num>(
        x: Self::Tensor<D>,
        lo: Self::Tensor<D>,
        hi: Self::Tensor<D>,
    ) -> Self::Tensor<D>;

    fn fma<D: ty::Float>(
        x: Self::Tensor<D>,
        y: Self::Tensor<D>,
        z: Self::Tensor<D>,
    ) -> Self::Tensor<D>;

    fn fdiv<D: ty::Float>(
        x: Self::Tensor<D>,
        y: Self::Tensor<D>,
        ieee_rounding: bool,
    ) -> Self::Tensor<D>;

    fn div_rn<D: ty::Float>(x: Self::Tensor<D>, y: Self::Tensor<D>) -> Self::Tensor<D>;

    fn umulhi(x: Self::Tensor<u32>, y: Self::Tensor<u32>) -> Self::Tensor<u32>;

    /// Ceiling integer division: `ceil(x / div)`.
    fn cdiv(x: i32, div: i32) -> i32;

    /// Swizzle 2-D indices for shared-memory bank-conflict avoidance.
    /// Returns the remapped `(i, j)` indices.
    fn swizzle2d(i: i32, j: i32, size_i: i32, size_j: i32, size_g: i32) -> (i32, i32);

    /*------------------------------ Reduction Ops ------------------------------*/

    /// Sum all elements along `axis`. `axis = None` reduces all dimensions.
    fn sum<D: ty::Num>(x: Self::Tensor<D>, axis: Option<i32>, keep_dims: bool) -> Self::Tensor<D>;

    /// Maximum along `axis`. `axis = None` reduces all dimensions.
    fn max<D: ty::Num>(x: Self::Tensor<D>, axis: Option<i32>, keep_dims: bool) -> Self::Tensor<D>;

    /// Maximum along `axis`, also returning the index of the maximum.
    ///
    /// - `tie_break_left`: when `true`, the leftmost index wins on ties (default `true`).
    fn max_with_indices<D: ty::Num>(
        x: Self::Tensor<D>,
        axis: i32,
        tie_break_left: bool,
        keep_dims: bool,
    ) -> (Self::Tensor<D>, Self::I32Tensor);

    /// Minimum along `axis`. `axis = None` reduces all dimensions.
    fn min<D: ty::Num>(x: Self::Tensor<D>, axis: Option<i32>, keep_dims: bool) -> Self::Tensor<D>;

    /// Minimum along `axis`, also returning the index of the minimum.
    ///
    /// - `tie_break_left`: when `true`, the leftmost index wins on ties (default `true`).
    fn min_with_indices<D: ty::Num>(
        x: Self::Tensor<D>,
        axis: i32,
        tie_break_left: bool,
        keep_dims: bool,
    ) -> (Self::Tensor<D>, Self::I32Tensor);

    /// Index of the maximum along `axis`.
    ///
    /// - `tie_break_left`: when `true`, the leftmost index wins on ties (default `true`).
    fn argmax<D: ty::Num>(
        x: Self::Tensor<D>,
        axis: i32,
        tie_break_left: bool,
        keep_dims: bool,
    ) -> Self::I32Tensor;

    /// Index of the minimum along `axis`.
    ///
    /// - `tie_break_left`: when `true`, the leftmost index wins on ties (default `true`).
    fn argmin<D: ty::Num>(
        x: Self::Tensor<D>,
        axis: i32,
        tie_break_left: bool,
        keep_dims: bool,
    ) -> Self::I32Tensor;

    /// XOR-reduction along `axis`. `axis = None` reduces all dimensions.
    fn xor_sum<D: ty::Int>(
        x: Self::Tensor<D>,
        axis: Option<i32>,
        keep_dims: bool,
    ) -> Self::Tensor<D>;

    /*------------------------------ Scan / Sort Ops ------------------------------*/

    fn cumsum<D: ty::Num>(x: Self::Tensor<D>, axis: i32, reverse: bool) -> Self::Tensor<D>;

    fn cumprod<D: ty::Num>(x: Self::Tensor<D>, axis: i32, reverse: bool) -> Self::Tensor<D>;

    /// Sort along `dim`. `dim = None` sorts along the last dimension.
    fn sort<D: ty::Num>(x: Self::Tensor<D>, dim: Option<i32>, descending: bool) -> Self::Tensor<D>;

    /// Compute a histogram with `num_bins` bins (width 1, starting at 0).
    ///
    /// - `mask`: when `Some`, masked-off elements are excluded (default `None`).
    fn histogram(
        x: Self::I32Tensor,
        num_bins: i32,
        mask: Option<Self::BoolTensor>,
    ) -> Self::I32Tensor;

    /// Generic reduction along `axis` using a user-supplied combine function.
    ///
    /// `combine_fn` must be a statically-known function pointer (corresponds to a
    /// `@triton.jit`-decorated helper in Python Triton).
    fn reduce<D: ty::Dtype, O: ty::Dtype>(
        x: Self::Tensor<D>,
        axis: i32,
        combine_fn: fn(Self::Tensor<O>, Self::Tensor<O>) -> Self::Tensor<O>,
        keep_dims: bool,
    ) -> Self::Tensor<O>;

    /// Generic prefix-scan along `axis` using a user-supplied combine function.
    ///
    /// - `reverse`: scan in the reverse direction (default `false`).
    fn associative_scan<D: ty::Dtype>(
        x: Self::Tensor<D>,
        axis: i32,
        combine_fn: fn(Self::Tensor<D>, Self::Tensor<D>) -> Self::Tensor<D>,
        reverse: bool,
    ) -> Self::Tensor<D>;

    /*------------------------------ Atomic Ops ------------------------------*/

    /// Atomic add. Returns the previous value.
    ///
    /// - `mask`: when `Some`, only masked lanes perform the operation (default `None`).
    /// - `sem`: memory ordering semantics (default `None` = AcqRel).
    /// - `scope`: synchronization scope (default `None` = Gpu).
    fn atomic_add<D: ty::Num>(
        ptr: Self::Tensor<Self::Pointer<D>>,
        val: Self::Tensor<D>,
        mask: Option<Self::BoolTensor>,
        sem: Option<MemSem>,
        scope: Option<MemScope>,
    ) -> Self::Tensor<D>;

    fn atomic_and<D: ty::Int>(
        ptr: Self::Tensor<Self::Pointer<D>>,
        val: Self::Tensor<D>,
        mask: Option<Self::BoolTensor>,
        sem: Option<MemSem>,
        scope: Option<MemScope>,
    ) -> Self::Tensor<D>;

    fn atomic_or<D: ty::Int>(
        ptr: Self::Tensor<Self::Pointer<D>>,
        val: Self::Tensor<D>,
        mask: Option<Self::BoolTensor>,
        sem: Option<MemSem>,
        scope: Option<MemScope>,
    ) -> Self::Tensor<D>;

    fn atomic_xor<D: ty::Int>(
        ptr: Self::Tensor<Self::Pointer<D>>,
        val: Self::Tensor<D>,
        mask: Option<Self::BoolTensor>,
        sem: Option<MemSem>,
        scope: Option<MemScope>,
    ) -> Self::Tensor<D>;

    fn atomic_max<D: ty::Num>(
        ptr: Self::Tensor<Self::Pointer<D>>,
        val: Self::Tensor<D>,
        mask: Option<Self::BoolTensor>,
        sem: Option<MemSem>,
        scope: Option<MemScope>,
    ) -> Self::Tensor<D>;

    fn atomic_min<D: ty::Num>(
        ptr: Self::Tensor<Self::Pointer<D>>,
        val: Self::Tensor<D>,
        mask: Option<Self::BoolTensor>,
        sem: Option<MemSem>,
        scope: Option<MemScope>,
    ) -> Self::Tensor<D>;

    fn atomic_xchg<D: ty::Dtype>(
        ptr: Self::Tensor<Self::Pointer<D>>,
        val: Self::Tensor<D>,
        mask: Option<Self::BoolTensor>,
        sem: Option<MemSem>,
        scope: Option<MemScope>,
    ) -> Self::Tensor<D>;

    /// Atomic compare-and-swap. Returns the previous value.
    fn atomic_cas<D: ty::Dtype>(
        ptr: Self::Tensor<Self::Pointer<D>>,
        cmp: Self::Tensor<D>,
        val: Self::Tensor<D>,
        sem: Option<MemSem>,
        scope: Option<MemScope>,
    ) -> Self::Tensor<D>;

    /*------------------------------ Random Number Generation ------------------------------*/

    /// Uniform random `f32` in `[0, 1)`.
    ///
    /// - `n_rounds`: number of Philox rounds (default `10`).
    fn rand(seed: u32, offsets: Self::I32Tensor, n_rounds: i32) -> Self::Tensor<f32>;

    /// Standard-normal random `f32`.
    ///
    /// - `n_rounds`: number of Philox rounds (default `10`).
    fn randn(seed: u32, offsets: Self::I32Tensor, n_rounds: i32) -> Self::Tensor<f32>;

    /// Random `i32`.
    ///
    /// - `n_rounds`: number of Philox rounds (default `10`).
    fn randint(seed: u32, offsets: Self::I32Tensor, n_rounds: i32) -> Self::I32Tensor;

    /// Four random `i32` streams (maximally efficient Philox entry point).
    ///
    /// - `n_rounds`: number of Philox rounds (default `10`).
    fn randint4x(
        seed: u32,
        offsets: Self::I32Tensor,
        n_rounds: i32,
    ) -> (
        Self::I32Tensor,
        Self::I32Tensor,
        Self::I32Tensor,
        Self::I32Tensor,
    );

    /*------------------------------ Inline Assembly ------------------------------*/

    /// Emit inline PTX/assembly applied element-wise across a tensor.
    ///
    /// - `asm`: the assembly template string.
    /// - `constraints`: register constraint string.
    /// - `is_pure`: whether the assembly has no side-effects (may be CSE'd).
    /// - `pack`: number of elements packed into each register.
    fn inline_asm_elementwise<D: ty::Dtype>(
        asm: &str,
        constraints: &str,
        is_pure: bool,
        pack: i32,
    ) -> Self::Tensor<D>;

    /*------------------------------ Compiler Hint Ops ------------------------------*/

    /// Assert that `cond` is always true, allowing the compiler to assume so.
    fn assume(cond: Self::BoolTensor);

    /// Hint that values of `x` are always multiples of the given constants.
    fn multiple_of<D: ty::Dtype>(x: Self::Tensor<D>, values: &[i32]) -> Self::Tensor<D>;

    /// Hint that `x` has `values[i]` contiguous elements along dimension `i`.
    fn max_contiguous<D: ty::Dtype>(x: Self::Tensor<D>, values: &[i32]) -> Self::Tensor<D>;

    /// Hint that `x` has `values[i]` constant elements along dimension `i`.
    fn max_constancy<D: ty::Dtype>(x: Self::Tensor<D>, values: &[i32]) -> Self::Tensor<D>;

    /*------------------------------ Debug Ops ------------------------------*/

    /// Insert a memory barrier for debugging purposes.
    fn debug_barrier();

    /// Emit a runtime assertion on the device. No-op when `cond` is `true`.
    ///
    /// - `msg`: message shown on assertion failure (default `""`).
    /// - `mask`: when `Some`, only lanes where mask is `true` check the assertion.
    fn device_assert(cond: Self::BoolTensor, msg: &str, mask: Option<Self::BoolTensor>);

    /// Print a tensor value from device code for debugging.
    ///
    /// - `hex`: print values in hexadecimal (default `false`).
    fn device_print<D: ty::Dtype>(prefix: &str, val: Self::Tensor<D>, hex: bool);

    /// Compile-time assertion (evaluated before kernel launch).
    fn static_assert(cond: bool, msg: &str);

    /// Compile-time print (evaluated before kernel launch).
    fn static_print(msg: &str);
}
