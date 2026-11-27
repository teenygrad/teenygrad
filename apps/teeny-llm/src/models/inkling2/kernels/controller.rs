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

//! Inkling-2's loop controller (`Inkling2LoopMessageGate`) in two kernels:
//!
//! ```text
//! g   = down(cat(norm(hidden), norm(message)))                 ControllerDownOp
//! out = (1 - tanh(forget(g))) * hidden + tanh(gain(g)) * message  ControllerApplyOp
//! ```
//!
//! It was nine: two norms, a concat, three linears, two tanh and the mix, every one a
//! launch on a tensor of a few thousand values. A core-layer execution has four
//! controllers, so they were about 1,300 launches a token, and the launches, not the
//! arithmetic, were their cost.
//!
//! The weights are bf16, packed two to a word as [`crate::kernels::linear`] packs
//! them: word `j` of a row holds column `j` low and column `j + K / 2` high. For
//! `down`, whose input is `cat(norm(hidden), norm(message))`, that pairs `hidden[j]`
//! with `message[j]`, so one pass reads both, and accumulates both norms' sums of
//! squares, without the concatenation existing.
//!
//! The gates are computed from the normalised inputs but applied to the raw ones, as
//! [`super::mix`] describes.

use teeny_core::dtype::{AddOffsets, Comparison, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// `g[t, r] = down[r] . cat(norm(h[t]), norm(m[t]))`, four outputs `r` per program.
///
/// Grid: `[RANK / 4, T]`.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn controller_down_forward<T: Triton, D: Float, const BLOCK: i32>(
    h_ptr: InPtr<T::Pointer<D>>,
    m_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<D>>,
    g_ptr: OutPtr<T::Pointer<D>>,
    HID: i32,
    RANK: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::BitAnd<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Shl<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let r = T::program_id(Axis::X) * 4;
    let t = T::program_id(Axis::Y);
    let offs = T::arange(0, BLOCK);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let mut sh = T::zeros::<D>(&[BLOCK]);
    let mut sm = T::zeros::<D>(&[BLOCK]);
    let mut h0 = T::zeros::<D>(&[BLOCK]);
    let mut h1 = T::zeros::<D>(&[BLOCK]);
    let mut h2 = T::zeros::<D>(&[BLOCK]);
    let mut h3 = T::zeros::<D>(&[BLOCK]);
    let mut m0 = T::zeros::<D>(&[BLOCK]);
    let mut m1 = T::zeros::<D>(&[BLOCK]);
    let mut m2 = T::zeros::<D>(&[BLOCK]);
    let mut m3 = T::zeros::<D>(&[BLOCK]);
    let n_chunks = (HID + BLOCK - 1) / BLOCK;
    for i in 0..n_chunks {
        let j = offs + i * BLOCK;
        let jm = j.lt(HID);
        let h = T::load(h_ptr.add_offsets(j + t * HID), Some(jm), Some(zeros), &[], None, None, None, false);
        let m = T::load(m_ptr.add_offsets(j + t * HID), Some(jm), Some(zeros), &[], None, None, None, false);
        sh = sh + h * h;
        sm = sm + m * m;
        let w0 = T::cast::<D, i32>(
            T::load(w_ptr.add_offsets(j + r * HID), Some(jm), Some(zeros), &[], None, None, None, false),
            None,
            true,
        );
        h0 = h0 + h * T::cast::<i32, D>(w0 << 16, None, true);
        m0 = m0 + m * T::cast::<i32, D>(w0 & -65536, None, true);
        let w1 = T::cast::<D, i32>(
            T::load(w_ptr.add_offsets(j + (r + 1) * HID), Some(jm), Some(zeros), &[], None, None, None, false),
            None,
            true,
        );
        h1 = h1 + h * T::cast::<i32, D>(w1 << 16, None, true);
        m1 = m1 + m * T::cast::<i32, D>(w1 & -65536, None, true);
        let w2 = T::cast::<D, i32>(
            T::load(w_ptr.add_offsets(j + (r + 2) * HID), Some(jm), Some(zeros), &[], None, None, None, false),
            None,
            true,
        );
        h2 = h2 + h * T::cast::<i32, D>(w2 << 16, None, true);
        m2 = m2 + m * T::cast::<i32, D>(w2 & -65536, None, true);
        let w3 = T::cast::<D, i32>(
            T::load(w_ptr.add_offsets(j + (r + 3) * HID), Some(jm), Some(zeros), &[], None, None, None, false),
            None,
            true,
        );
        h3 = h3 + h * T::cast::<i32, D>(w3 << 16, None, true);
        m3 = m3 + m * T::cast::<i32, D>(w3 & -65536, None, true);
    }
    // norm(x) = x / sqrt(mean(x^2) + 1e-6), a scalar per token, so it factors out of
    // each dot product.
    let n = T::cast::<i32, D>(T::zeros::<i32>(&[BLOCK]) + HID, None, false);
    let eps = T::full(&[BLOCK], D::from_f64(1e-6));
    let rh = T::rsqrt(T::sum(sh, Some(0), true) / n + eps);
    let rm = T::rsqrt(T::sum(sm, Some(0), true) / n + eps);
    let one = offs.lt(1);
    let base = t * RANK + r;
    T::store(
        g_ptr.add_offsets(offs + base),
        T::sum(h0, Some(0), true) * rh + T::sum(m0, Some(0), true) * rm + zeros,
        Some(one),
        &[],
        None,
        None,
    );
    T::store(
        g_ptr.add_offsets(offs + (base + 1)),
        T::sum(h1, Some(0), true) * rh + T::sum(m1, Some(0), true) * rm + zeros,
        Some(one),
        &[],
        None,
        None,
    );
    T::store(
        g_ptr.add_offsets(offs + (base + 2)),
        T::sum(h2, Some(0), true) * rh + T::sum(m2, Some(0), true) * rm + zeros,
        Some(one),
        &[],
        None,
        None,
    );
    T::store(
        g_ptr.add_offsets(offs + (base + 3)),
        T::sum(h3, Some(0), true) * rh + T::sum(m3, Some(0), true) * rm + zeros,
        Some(one),
        &[],
        None,
        None,
    );
}

/// `out[t, c] = (1 - tanh(forget[c] . g[t])) * h[t, c] + tanh(gain[c] . g[t]) * m[t, c]`
/// for `BLOCK` channels `c` per program.
///
/// Grid: `[ceil(HID / BLOCK), T]`.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn controller_apply_forward<T: Triton, D: Float, const BLOCK: i32>(
    h_ptr: InPtr<T::Pointer<D>>,
    m_ptr: InPtr<T::Pointer<D>>,
    g_ptr: InPtr<T::Pointer<D>>,
    wf_ptr: InPtr<T::Pointer<D>>,
    wg_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    HID: i32,
    RANK: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Mul<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::BitAnd<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Shl<i32, Output = T::I32Tensor>,
    T::I32Tensor: Comparison<i32, BoolTensor = T::BoolTensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
{
    let c = T::arange(0, BLOCK) + T::program_id(Axis::X) * BLOCK;
    let t = T::program_id(Axis::Y);
    let cm = c.lt(HID);
    let zeros = T::zeros::<D>(&[BLOCK]);
    let half = RANK / 2;
    let mut f = T::zeros::<D>(&[BLOCK]);
    let mut a = T::zeros::<D>(&[BLOCK]);
    for j in 0..half {
        // g[t, j] and g[t, j + RANK / 2], the two columns word j of a row holds.
        let g0 = T::load(
            g_ptr.add_offsets(T::zeros::<i32>(&[BLOCK]) + (t * RANK + j)),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let g1 = T::load(
            g_ptr.add_offsets(T::zeros::<i32>(&[BLOCK]) + (t * RANK + j + half)),
            None,
            None,
            &[],
            None,
            None,
            None,
            false,
        );
        let wf = T::cast::<D, i32>(
            T::load(wf_ptr.add_offsets(c * half + j), Some(cm), Some(zeros), &[], None, None, None, false),
            None,
            true,
        );
        f = f + T::cast::<i32, D>(wf << 16, None, true) * g0 + T::cast::<i32, D>(wf & -65536, None, true) * g1;
        let wg = T::cast::<D, i32>(
            T::load(wg_ptr.add_offsets(c * half + j), Some(cm), Some(zeros), &[], None, None, None, false),
            None,
            true,
        );
        a = a + T::cast::<i32, D>(wg << 16, None, true) * g0 + T::cast::<i32, D>(wg & -65536, None, true) * g1;
    }

    // tanh(x) = 2 / (1 + exp(-2x)) - 1, with exp(-2x) in lane as softcap computes
    // it (range reduction to 2^n and a degree-6 polynomial, the argument clamped to
    // [-87, 88]); `gated_mul` documents the method.
    let one = T::full(&[BLOCK], D::from_f64(1.0));
    let two = T::full(&[BLOCK], D::from_f64(2.0));
    let lo = T::full(&[BLOCK], D::from_f64(-87.0));
    let hiclamp = T::full(&[BLOCK], D::from_f64(88.0));
    let log2e = T::full(&[BLOCK], D::from_f64(1.442_695_040_888_963_4));
    let magic = T::full(&[BLOCK], D::from_f64(12_582_912.0)); // 1.5 * 2^23
    let c0 = T::full(&[BLOCK], D::from_f64(0.000_154_035_303_933_816_09));
    let c1 = T::full(&[BLOCK], D::from_f64(0.001_333_355_814_642_844_3));
    let c2 = T::full(&[BLOCK], D::from_f64(0.009_618_129_107_628_477));
    let c3 = T::full(&[BLOCK], D::from_f64(0.055_504_108_664_821_58));
    let c4 = T::full(&[BLOCK], D::from_f64(0.240_226_506_959_100_7));
    let c5 = T::full(&[BLOCK], D::from_f64(0.693_147_180_559_945_3));

    let af = zeros - two * f;
    let af = T::where_(T::lt(af, lo), lo, af);
    let af = T::where_(T::gt(af, hiclamp), hiclamp, af);
    let yf = af * log2e;
    let rf = (yf + magic) - magic;
    let ff = yf - rf;
    let pf = ((((((c0 * ff + c1) * ff + c2) * ff + c3) * ff + c4) * ff + c5) * ff) + one;
    let nf = T::cast::<D, i32>(rf, None, false);
    let ef = pf * T::cast::<i32, D>((nf + 127) * 8_388_608, None, true);
    let tf = two / (one + ef) - one;

    let aa = zeros - two * a;
    let aa = T::where_(T::lt(aa, lo), lo, aa);
    let aa = T::where_(T::gt(aa, hiclamp), hiclamp, aa);
    let ya = aa * log2e;
    let ra = (ya + magic) - magic;
    let fa = ya - ra;
    let pa = ((((((c0 * fa + c1) * fa + c2) * fa + c3) * fa + c4) * fa + c5) * fa) + one;
    let na = T::cast::<D, i32>(ra, None, false);
    let ea = pa * T::cast::<i32, D>((na + 127) * 8_388_608, None, true);
    let ta = two / (one + ea) - one;

    let h = T::load(h_ptr.add_offsets(c + t * HID), Some(cm), Some(zeros), &[], None, None, None, false);
    let m = T::load(m_ptr.add_offsets(c + t * HID), Some(cm), Some(zeros), &[], None, None, None, false);
    T::store(
        y_ptr.add_offsets(c + t * HID),
        (one - tf) * h + ta * m,
        Some(cm),
        &[],
        None,
        None,
    );
}

// ---------------------------------------------------------------------------
// Graph-level ops
// ---------------------------------------------------------------------------

use core::marker::PhantomData;
use std::any::Any;
use std::sync::Arc;
use teeny_core::device::program::ArgVisitor;
use teeny_core::dtype::FloatBytes;
use teeny_core::graph::{CustomOp, Shape};
use teeny_core::model::{RawPtr, RuntimeOp};

/// Lanes along `hidden` for the down projection: ten accumulators live at once, so
/// 32 f32 lanes (one register each at VLEN=256) keep them in registers.
const DOWN_BLOCK: i32 = 32;

/// Output channels per apply program. At 1024 every vector is the whole register
/// file and the tanh spilled 1,400 times; 64 keeps it in registers.
const APPLY_BLOCK: i32 = 64;

/// `g = down(cat(norm(hidden), norm(message)))`: inputs `hidden` and `message`, each
/// `[T, hidden]`; output `[T, rank]`. Weight slot `down.weight_bf16`, `[rank, hidden]`
/// words, packed by [`crate::kernels::linear::pack_bf16_halves`] from `[rank, 2 * hidden]`.
pub struct ControllerDownOp<D: FloatBytes + Send + Sync + 'static> {
    /// `hidden`.
    pub hid: usize,
    /// `gate_rank`, a multiple of 4.
    pub rank: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> ControllerDownOp<D> {
    /// Creates the op.
    pub fn new(hid: usize, rank: usize) -> Self {
        assert!(rank.is_multiple_of(4), "controller down: rank {rank} is not a multiple of 4");
        Self { hid, rank, _phantom: PhantomData }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for ControllerDownOp<D> {
    fn name(&self) -> &str {
        "llm.controller_down"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[0][0], Some(self.rank)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = ControllerDownForward::<D>::new(DOWN_BLOCK);
        Some((
            "controller_down_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(ControllerDownRuntimeOp::<D> { hid: self.hid, rank: self.rank, _phantom: PhantomData }),
        ))
    }
}

/// Runtime dispatch for [`ControllerDownOp`].
pub struct ControllerDownRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    hid: usize,
    rank: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for ControllerDownRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        2
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![self.rank, self.hid]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["down.weight_bf16"]
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
        assert_eq!(inputs[0].1[1], self.hid, "controller down: hidden width");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.hid as i32);
        visitor.visit_i32(self.rank as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [(self.rank / 4) as u32, output_shape[0] as u32, 1]
    }
}

/// `(1 - tanh(forget(g))) * hidden + tanh(gain(g)) * message`: inputs `hidden`,
/// `message` (`[T, hidden]`) and `g` (`[T, rank]`); output `[T, hidden]`. Weight slots
/// `forget.weight_bf16` and `gain.weight_bf16`, `[hidden, rank / 2]` words each.
pub struct ControllerApplyOp<D: FloatBytes + Send + Sync + 'static> {
    /// `hidden`.
    pub hid: usize,
    /// `gate_rank`, even.
    pub rank: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> ControllerApplyOp<D> {
    /// Creates the op.
    pub fn new(hid: usize, rank: usize) -> Self {
        assert!(rank.is_multiple_of(2), "controller apply: rank {rank} is odd");
        Self { hid, rank, _phantom: PhantomData }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for ControllerApplyOp<D> {
    fn name(&self) -> &str {
        "llm.controller_apply"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        input_shapes[0].to_vec()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = ControllerApplyForward::<D>::new(APPLY_BLOCK);
        Some((
            "controller_apply_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(ControllerApplyRuntimeOp::<D> { hid: self.hid, rank: self.rank, _phantom: PhantomData }),
        ))
    }
}

/// Runtime dispatch for [`ControllerApplyOp`].
pub struct ControllerApplyRuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    hid: usize,
    rank: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for ControllerApplyRuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        3
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![self.hid, self.rank / 2], vec![self.hid, self.rank / 2]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["forget.weight_bf16", "gain.weight_bf16"]
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
        assert_eq!(inputs[0].1[1], self.hid, "controller apply: hidden width");
        assert_eq!(inputs[2].1[1], self.rank, "controller apply: rank");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(inputs[1].0);
        visitor.visit_ptr(inputs[2].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(params[1]);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.hid as i32);
        visitor.visit_i32(self.rank as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [self.hid.div_ceil(APPLY_BLOCK as usize) as u32, output_shape[0] as u32, 1]
    }
}
