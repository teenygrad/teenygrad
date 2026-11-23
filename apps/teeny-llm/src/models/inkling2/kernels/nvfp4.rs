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

//! NVFP4 dequantisation, as the expert gather needs it.
//!
//! The experts of `thinkingmachines/Inkling-2-Mini` ship as NVFP4: an E2M1 nibble per
//! weight, an E4M3 scale per 16 of them, and one f32 scale per expert.
//!
//! ```text
//! value = e2m1(nibble) * e4m3(scale[k / 16]) * scale2[expert]
//! ```
//!
//! Dequantising at load time is not an option: the experts are 20.2 GB packed, which
//! becomes about 160 GB in f32, against the 31 GB a K3 board has. They have to stay
//! packed in memory and be unpacked per block at the multiply-add, which is what this
//! does.
//!
//! # The interim encoding, and what it costs
//!
//! Parameter buffers take the graph's dtype, so a packed byte cannot simply be a
//! `u8` element. Two encodings are possible:
//!
//! - **One packed byte per f32 element**, as a value in `0 ..= 255`. Extraction is
//!   `lo = b - (b / 16) * 16` and `hi = b / 16`, pure integer arithmetic. This is
//!   what the kernel below does. It costs 4 bytes per 2 weights, so 2 bytes per
//!   weight: better than f32's 4, but it leaves Inkling-2-Mini's experts at about
//!   40 GB, which still does not fit a board.
//! - **Four packed bytes per f32 element**, bit-packed, which would be the true
//!   0.5 bytes per weight. `T::cast` can bitcast an f32 to an i32, but extracting
//!   bytes from it needs a logical shift and a mask, and the DSL has **neither**:
//!   `LlvmI32Tensor` implements `Add`, `Sub`, `Mul` and `Div` by a scalar and nothing
//!   else, and `Div` truncates towards zero, so a word whose top bit is set cannot be
//!   taken apart arithmetically.
//!
//! So the arithmetic here is final and the addressing is not. Adding `Shr` and
//! `BitAnd` for `LlvmI32Tensor` to teeny-triton turns this kernel into the packed
//! one by changing how `byte` is fetched, and nothing else.
//!
//! The decode itself is the same one measured in `experiments/nvfp4-riscv`, and the
//! tests here run against the same real weight bytes.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `y[i] = e2m1(nibble i) * e4m3(scale[i / 16]) * scale2`.
///
/// `b_ptr` holds one packed byte per element as a value `0 ..= 255`, two weights
/// each; `s_ptr` one E4M3 scale byte per element, likewise as a value.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn nvfp4_dequant_forward<T: Triton, D: Float, const BLOCK: i32>(
    b_ptr: InPtr<T::Pointer<D>>,
    s_ptr: InPtr<T::Pointer<D>>,
    s2_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    N: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Div<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<T::I32Tensor, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let i = T::arange(0, BLOCK) + pid * BLOCK;
    let m = i.lt(N);
    let zeros = T::zeros::<D>(&[BLOCK]);

    // Two weights per byte, low nibble first.
    let byte_idx = i / 2;
    let is_high = i - (i / 2) * 2;
    let b = T::cast::<D, i32>(
        T::load(
            b_ptr.add_offsets(byte_idx),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        ),
        None,
        false,
    );
    let hi = b / 16;
    let lo = b - hi * 16;
    let nib = T::where_(is_high.eq(1), hi, lo);

    // E2M1: s:1 e:2 m:1 over 0, 0.5, 1, 1.5, 2, 3, 4, 6.
    let sign_bit = nib / 8;
    let rest = nib - sign_bit * 8;
    let e = rest / 2;
    let mant = rest - e * 2;
    let mf = T::cast::<i32, D>(mant, None, false);
    let half = T::full(&[BLOCK], D::from_f64(0.5));
    let one = T::full(&[BLOCK], D::from_f64(1.0));
    // e == 0 is the subnormal pair {0, 0.5}; otherwise (1 + m/2) * 2^(e-1).
    let ef = T::cast::<i32, D>(e, None, false);
    let mag = T::where_(e.eq(0), mf * half, (one + mf * half) * T::exp2(ef - one));
    let signed = T::where_(sign_bit.eq(1), zeros - mag, mag);

    // E4M3 block scale, one per 16 weights.
    let sb = T::cast::<D, i32>(
        T::load(
            s_ptr.add_offsets(i / 16),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        ),
        None,
        false,
    );
    let ssign = sb / 128;
    let srest = sb - ssign * 128;
    let se = srest / 8;
    let sm = srest - se * 8;
    let smf = T::cast::<i32, D>(sm, None, false);
    let eighth = T::full(&[BLOCK], D::from_f64(0.125));
    let seven = T::full(&[BLOCK], D::from_f64(7.0));
    let sef = T::cast::<i32, D>(se, None, false);
    // Normal: (1 + m/8) * 2^(e-7). Subnormal (e == 0): (m/8) * 2^-6.
    let sval = T::where_(
        se.eq(0),
        smf * eighth * T::full(&[BLOCK], D::from_f64(0.015_625)),
        (one + smf * eighth) * T::exp2(sef - seven),
    );
    let sval = T::where_(ssign.eq(1), zeros - sval, sval);

    let s2 = T::load(
        s2_ptr.add_offsets(T::zeros::<i32>(&[BLOCK])),
        None,
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    T::store(
        y_ptr.add_offsets(i),
        signed * sval * s2,
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

const BLOCK: i32 = 256;

/// Graph-level NVFP4 dequantisation of one expert plane.
///
/// Inputs are `bytes` (one packed byte per element, two weights each) and `scales`
/// (one E4M3 byte per element, one per 16 weights); the parameter is the expert's f32
/// `scale2`. The output is `n` dequantised weights.
pub struct Nvfp4DequantOp<D: FloatBytes + Send + Sync + 'static> {
    /// Weights produced.
    pub n: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> Nvfp4DequantOp<D> {
    /// A dequantisation producing `n` weights.
    pub fn new(n: usize) -> Self {
        assert!(
            n.is_multiple_of(16),
            "nvfp4: {n} weights is not a whole number of 16-blocks"
        );
        Self {
            n,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for Nvfp4DequantOp<D> {
    fn name(&self) -> &str {
        "llm.nvfp4_dequant"
    }

    fn infer_output_shape(&self, _input_shapes: &[&Shape]) -> Shape {
        vec![Some(1), Some(self.n)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = Nvfp4DequantForward::<D>::new(BLOCK);
        Some((
            "nvfp4_dequant_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(Nvfp4DequantRuntimeOp::<D> {
                n: self.n,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`Nvfp4DequantOp`].
pub struct Nvfp4DequantRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    n: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for Nvfp4DequantRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        2
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![1]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["scale2"]
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.n as i32);
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

/// Extracts byte `WHICH` of a bit-packed word, which is how the packed NVFP4
/// encoding is unpacked.
///
/// `x` is bitcast to an i32 and one of its four bytes is returned. There is no shift:
/// `Shr` is implemented for `LlvmI32Tensor` but **teenyc's MLIR backend does not
/// lower it** (it yields an unshaped `tensor<?xf32>` and the store then fails to
/// verify), whereas `BitAnd` lowers correctly. Masking first and dividing second
/// gives the same answer and keeps every intermediate non-negative, which matters
/// because `Div` truncates towards zero.
///
/// The top byte carries the sign bit, so it is masked to seven bits and 128 added
/// back when the word is negative.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn unpack_byte_forward<T: Triton, D: Float, const BLOCK: i32, const WHICH: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    N: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Div<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::BitAnd<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let pid = T::program_id(Axis::X);
    let i = T::arange(0, BLOCK) + pid * BLOCK;
    let m = i.lt(N);
    let zeros = T::zeros::<D>(&[BLOCK]);
    // Bitcast, not a value conversion: the f32 is a container for four bytes.
    let w = T::cast::<D, i32>(
        T::load(
            x_ptr.add_offsets(i),
            Some(m),
            Some(zeros),
            &[],
            None,
            None,
            None,
            false,
        ),
        None,
        true,
    );
    let byte = if WHICH == 0 {
        w & 0xFF
    } else if WHICH == 1 {
        (w & 0xFF00) / 256
    } else if WHICH == 2 {
        (w & 0x00FF_0000) / 65536
    } else {
        // Bits 24..31, with bit 31 put back by hand.
        let low7 = (w & 0x7F00_0000) / 16_777_216;
        T::where_(w.lt(0), low7 + 128, low7 + 0)
    };
    T::store(
        y_ptr.add_offsets(i),
        T::cast::<i32, D>(byte, None, false),
        Some(m),
        &[],
        None,
        None,
    );
}

/// Graph-level byte extraction from a bit-packed f32: `y = byte WHICH of x`.
pub struct UnpackByteOp<D: FloatBytes + Send + Sync + 'static> {
    /// Which byte, 0 (lowest) to 3 (the one holding the sign bit).
    pub which: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> UnpackByteOp<D> {
    /// Extracts byte `which` of each word.
    pub fn new(which: usize) -> Self {
        assert!(which < 4, "unpack: byte {which} is outside a 32-bit word");
        Self {
            which,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for UnpackByteOp<D> {
    fn name(&self) -> &str {
        "llm.unpack_byte"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].to_vec()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = UnpackByteForward::<D>::new(BLOCK, self.which as i32);
        Some((
            "unpack_byte_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(UnpackByteRuntimeOp::<D> {
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`UnpackByteOp`].
pub struct UnpackByteRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for UnpackByteRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
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
