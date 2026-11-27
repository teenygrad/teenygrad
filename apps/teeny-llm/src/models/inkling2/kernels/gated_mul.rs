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

//! The sigmoid gate of Inkling-2's output projection
//! (`Inkling2LoopGatedOutputProjection`), elementwise over two tensors of one shape:
//!
//! ```text
//! y = x * sigmoid(g)
//! ```
//!
//! `g` is the gate projection of the **layer input**, not of the attention output, so
//! the layer decides how much of the attention result to let through based on what it
//! was given. Both arguments are already `[T, heads * head_dim]`.
//!
//! Grid: `ceil(N / BLOCK)`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `y = x * sigmoid(g)`.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn gated_mul_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    g_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    N: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let offs = T::arange(0, BLOCK) + pid * BLOCK;
    let m = offs.lt(N);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let x = T::load(
        x_ptr.add_offsets(offs),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let g = T::load(
        g_ptr.add_offsets(offs),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let one = T::full(&[BLOCK], D::from_f64(1.0));

    // exp(-g) in lane, rather than `T::exp`.
    //
    // RVV has no exponential instruction, so `T::exp` lowers to one `expf` call per
    // lane. At BLOCK = 1024 that was this whole kernel: 1024 `call expf` in 7867 lines
    // of assembly, for two loads, a sigmoid and a store. So the usual range reduction,
    //
    //     e^x = 2^(x * log2 e) = 2^n * 2^f,   n integral, f in [-0.5, 0.5]
    //
    // with 2^f a degree-6 polynomial in f -- the coefficients are ln(2)^k / k!, the
    // Taylor series of 2^f; a minimax fit would buy accuracy we do not need -- and 2^n
    // built by putting n + 127 into an f32's exponent field. Two things that are easy
    // to get wrong:
    //
    //   - n comes from the round-to-nearest trick `(y + 1.5 * 2^23) - 1.5 * 2^23`, not
    //     `T::floor`, which is a libm call per lane too and would defeat the point. It
    //     is exact for |y| < 2^22 and y is at most ~127 here.
    //   - the input is clamped to [-87, 88] first. Past that the exponent field
    //     overflows and 2^n is a NaN or an infinity instead of saturating.
    //
    // ~1e-7 relative, against the 1e-4 the tests ask of the sigmoid. It is written out
    // here and in `swiglu` rather than shared because `#[kernel]` compiles the function
    // body on its own, with no access to anything else in the crate -- not a helper
    // function and not a macro.
    let lo = T::full(&[BLOCK], D::from_f64(-87.0));
    let hi = T::full(&[BLOCK], D::from_f64(88.0));
    let ex = zeros - g;
    let ex = T::where_(T::lt(ex, lo), lo, ex);
    let ex = T::where_(T::gt(ex, hi), hi, ex);

    let y = ex * T::full(&[BLOCK], D::from_f64(1.442_695_040_888_963_4));
    let magic = T::full(&[BLOCK], D::from_f64(12_582_912.0)); // 1.5 * 2^23
    let rounded = (y + magic) - magic;
    let f = y - rounded;

    let p = T::full(&[BLOCK], D::from_f64(0.000_154_035_303_933_816_09));
    let p = p * f + T::full(&[BLOCK], D::from_f64(0.001_333_355_814_642_844_3));
    let p = p * f + T::full(&[BLOCK], D::from_f64(0.009_618_129_107_628_477));
    let p = p * f + T::full(&[BLOCK], D::from_f64(0.055_504_108_664_821_58));
    let p = p * f + T::full(&[BLOCK], D::from_f64(0.240_226_506_959_100_7));
    let p = p * f + T::full(&[BLOCK], D::from_f64(0.693_147_180_559_945_3));
    let p = p * f + one;

    // `* 8388608` is `<< 23`: n + 127 into the exponent field.
    let n = T::cast::<D, i32>(rounded, None, false);
    let scale = T::cast::<i32, D>((n + 127) * 8_388_608, None, true);
    let eg = p * scale;

    T::store(
        y_ptr.add_offsets(offs),
        x * (one / (one + eg)),
        Some(m),
        &[],
        None,
        None,
    );
}

// ---------------------------------------------------------------------------
// Graph-level op
// ---------------------------------------------------------------------------

use core::marker::PhantomData;
use std::any::Any;
use std::sync::Arc;
use teeny_core::device::program::ArgVisitor;
use teeny_core::dtype::FloatBytes;
use teeny_core::graph::{CustomOp, Shape};
use teeny_core::model::{RawPtr, RuntimeOp};

// 128 lanes. With the exponential vectorised the tile no longer has to be large to
// hide a per-lane libm call, and 1024 lanes at `e32` with VLEN=256 is 128 vector
// registers: it spilled 199 times. 128 spills 10 and 64 only 2, but fewer lanes a
// launch means more launches, whose cost is not measured here, so this takes the knee.
const BLOCK: i32 = 128;

/// Graph-level `x * sigmoid(g)`; inputs `x` and `g` of one shape, no weights.
pub struct GatedMulOp<D: FloatBytes + Send + Sync + 'static> {
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> Default for GatedMulOp<D> {
    fn default() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for GatedMulOp<D> {
    fn name(&self) -> &str {
        "llm.gated_mul"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].to_vec()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = GatedMulForward::<D>::new(BLOCK);
        Some((
            "gated_mul_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(GatedMulRuntimeOp::<D> {
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`GatedMulOp`].
pub struct GatedMulRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for GatedMulRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        2
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        Vec::new()
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        _params: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(output_shape.iter().product::<usize>() as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [
            output_shape
                .iter()
                .product::<usize>()
                .div_ceil(BLOCK as usize) as u32,
            1,
            1,
        ]
    }
}
