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

//! Inkling-2's gated MLP activation (`Inkling2LoopSoftcappedSwiGLU`), fused:
//!
//! ```text
//! y = softcap(gate, gate_cap) * sigmoid(gate) * softcap(up, up_cap)
//! ```
//!
//! The expert's fused gate-and-up projection emits `[S, 2 * I]`; `chunk(2, -1)` in the
//! reference means the FIRST half is the gate and the second the up, so this reads
//! both halves of one row and emits `[S, I]`.
//!
//! The shape of it is worth stating, because it is not the usual SwiGLU: the sigmoid
//! sees the **raw** gate, while the softcaps bound only the linear gate factor and
//! `up`. Capping the gate before the sigmoid would squash the gate towards 0.5 and
//! change the activation everywhere, not just in the tails.
//!
//! Inkling-2-Mini uses 4.0 for the gate and 25.0 for the up. `tanh` and `sigmoid` are
//! both built from `exp`, as in Gemma's GELU. Grid: `ceil(I / BLOCK)` by `S`.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// The gated activation above.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn swiglu_forward<T: Triton, D: Float, const BLOCK: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    I: i32,
    GATE_CAP: f32,
    UP_CAP: f32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let bid = T::program_id(Axis::X);
    let row = T::program_id(Axis::Y);
    let offs = T::arange(0, BLOCK) + bid * BLOCK;
    let m = offs.lt(I);
    let zeros = T::zeros::<D>(&[BLOCK]);

    // One row of the projection is gate and up INTERLEAVED, not two halves: the
    // checkpoint stores `w13_weight` as gate, up, gate, up, ... down its rows, which
    // is why transformers converts it with an `Interleave(dim=1)` rather than a plain
    // rename. Splitting it into halves instead silently pairs each gate with the wrong
    // up, and the model still produces fluent-looking nonsense.
    let gate = T::load(
        x_ptr.add_offsets(offs * 2 + row * 2 * I),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );
    let up = T::load(
        x_ptr.add_offsets(offs * 2 + 1 + row * 2 * I),
        Some(m),
        Some(zeros),
        &[],
        None,
        None,
        None,
        false,
    );

    let one = T::full(&[BLOCK], D::from_f64(1.0));
    let two = T::full(&[BLOCK], D::from_f64(2.0));
    let gcap = T::full(&[BLOCK], D::from_f64(GATE_CAP as f64));
    let ucap = T::full(&[BLOCK], D::from_f64(UP_CAP as f64));

    // The three exponentials below are computed in lane. `T::exp` is one `expf` call
    // per lane on RVV, which made this kernel 768 calls in 6489 lines of assembly; the
    // reduction and its pitfalls are documented in `gated_mul`. Written out three times
    // because `#[kernel]` compiles a function body with no access to the rest of the
    // crate, so there is no helper to call and no macro to expand.
    let lo = T::full(&[BLOCK], D::from_f64(-87.0));
    let hi = T::full(&[BLOCK], D::from_f64(88.0));
    let log2e = T::full(&[BLOCK], D::from_f64(1.442_695_040_888_963_4));
    let magic = T::full(&[BLOCK], D::from_f64(12_582_912.0)); // 1.5 * 2^23
    let k6 = T::full(&[BLOCK], D::from_f64(0.000_154_035_303_933_816_09));
    let k5 = T::full(&[BLOCK], D::from_f64(0.001_333_355_814_642_844_3));
    let k4 = T::full(&[BLOCK], D::from_f64(0.009_618_129_107_628_477));
    let k3 = T::full(&[BLOCK], D::from_f64(0.055_504_108_664_821_58));
    let k2 = T::full(&[BLOCK], D::from_f64(0.240_226_506_959_100_7));
    let k1 = T::full(&[BLOCK], D::from_f64(0.693_147_180_559_945_3));

    // exp(-2 * gate / gcap)
    let a = zeros - two * (gate / gcap);
    let a = T::where_(T::lt(a, lo), lo, a);
    let a = T::where_(T::gt(a, hi), hi, a);
    let ya = a * log2e;
    let ra = (ya + magic) - magic;
    let fa = ya - ra;
    let pa = k6 * fa + k5;
    let pa = pa * fa + k4;
    let pa = pa * fa + k3;
    let pa = pa * fa + k2;
    let pa = pa * fa + k1;
    let pa = pa * fa + one;
    let na = T::cast::<D, i32>(ra, None, false);
    let ea = pa * T::cast::<i32, D>((na + 127) * 8_388_608, None, true);

    // exp(-2 * up / ucap)
    let b = zeros - two * (up / ucap);
    let b = T::where_(T::lt(b, lo), lo, b);
    let b = T::where_(T::gt(b, hi), hi, b);
    let yb = b * log2e;
    let rb = (yb + magic) - magic;
    let fb = yb - rb;
    let pb = k6 * fb + k5;
    let pb = pb * fb + k4;
    let pb = pb * fb + k3;
    let pb = pb * fb + k2;
    let pb = pb * fb + k1;
    let pb = pb * fb + one;
    let nb = T::cast::<D, i32>(rb, None, false);
    let eb = pb * T::cast::<i32, D>((nb + 127) * 8_388_608, None, true);

    // exp(-gate), for the sigmoid, which sees the raw gate and not the capped one.
    let c = zeros - gate;
    let c = T::where_(T::lt(c, lo), lo, c);
    let c = T::where_(T::gt(c, hi), hi, c);
    let yc = c * log2e;
    let rc = (yc + magic) - magic;
    let fc = yc - rc;
    let pc = k6 * fc + k5;
    let pc = pc * fc + k4;
    let pc = pc * fc + k3;
    let pc = pc * fc + k2;
    let pc = pc * fc + k1;
    let pc = pc * fc + one;
    let nc = T::cast::<D, i32>(rc, None, false);
    let ec = pc * T::cast::<i32, D>((nc + 127) * 8_388_608, None, true);

    // tanh(u) = 2 / (1 + exp(-2u)) - 1
    let gt = two / (one + ea) - one;
    let ut = two / (one + eb) - one;
    let sig = one / (one + ec);

    T::store(
        y_ptr.add_offsets(offs + row * I),
        gt * gcap * sig * (ut * ucap),
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

// 32 lanes. Three exponentials in flight means three times the live values of
// `gated_mul`, so the tile that fits the register file is smaller: 256 spilled 153
// times, 128 spills 49, 64 spills 15 and 32 none. The rule in both kernels is the
// largest tile whose spills are down in the noise, which lands on 128 there and 32
// here; which side of the launch-overhead trade that really sits on wants the board.
const BLOCK: i32 = 32;

/// Graph-level softcapped SwiGLU: `[S, 2 * I]` to `[S, I]`.
pub struct SwigluOp<D: FloatBytes + Send + Sync + 'static> {
    /// Width of one half of the projection.
    pub intermediate: usize,
    /// Softcap on the gate. 4.0 in Inkling-2-Mini.
    pub gate_cap: f32,
    /// Softcap on the up projection. 25.0 in Inkling-2-Mini.
    pub up_cap: f32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> SwigluOp<D> {
    /// A gated activation over `intermediate` lanes with the two caps.
    pub fn new(intermediate: usize, gate_cap: f32, up_cap: f32) -> Self {
        assert!(
            gate_cap > 0.0 && up_cap > 0.0,
            "swiglu: caps {gate_cap} and {up_cap} must be positive"
        );
        Self {
            intermediate,
            gate_cap,
            up_cap,
            _phantom: PhantomData,
        }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for SwigluOp<D> {
    fn name(&self) -> &str {
        "llm.swiglu_softcap"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[0][0], Some(self.intermediate)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = SwigluForward::<D>::new(BLOCK);
        Some((
            "swiglu_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(SwigluRuntimeOp::<D> {
                intermediate: self.intermediate,
                gate_cap: self.gate_cap,
                up_cap: self.up_cap,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`SwigluOp`].
pub struct SwigluRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    intermediate: usize,
    gate_cap: f32,
    up_cap: f32,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for SwigluRuntimeOp<D> {
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
        _output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        assert_eq!(
            inputs[0].1[1],
            2 * self.intermediate,
            "swiglu: the projection must be gate and up together"
        );
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.intermediate as i32);
        visitor.visit_f32(self.gate_cap);
        visitor.visit_f32(self.up_cap);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [
            self.intermediate.div_ceil(BLOCK as usize) as u32,
            output_shape[0] as u32,
            1,
        ]
    }
}
