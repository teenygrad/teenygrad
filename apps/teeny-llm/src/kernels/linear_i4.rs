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

//! Bias-free linears on int4 weights with a scale per group of input features, for
//! decode, which reads every linear weight once a token at memory bandwidth: half a
//! byte a weight plus a scale per [`I4_GROUP`] against int8's full byte.
//!
//! Output features are the lanes, as in [`super::linear::linear_lanes_i8_forward`]:
//! [`I4_LANES`] outputs a program, a batch of rows unpacking each weight once.

use teeny_core::dtype::{AddOffsets, Float, Tensor};
use teeny_macros::kernel;
use teeny_triton::triton::*;

/// Output features per program of [`linear_lanes_i4_forward`].
pub const I4_LANES: usize = 32;

/// Input features sharing one scale. Smaller groups follow the weights more
/// closely at more bytes for scales: 64 costs 1/16 byte a weight in f32 scales.
pub const I4_GROUP: usize = 64;

/// Rows of activations a program handles at once.
pub const ROW_TILE: usize = 4;

/// The name a converted checkpoint stores a linear's int4 buffer under. The lane
/// width and the group size are in the name because they are in the layout.
pub fn int4_name(weight: &str) -> String {
    format!("{weight}.i4g{I4_GROUP}x{I4_LANES}")
}

/// `x @ W^T` on int4 weights.
///
/// A weight is `q + 8` in a nibble (`q` in -7..=7): the byte at `[n / LANES][k / 2]
/// [n % LANES]` holds input feature `k` (even) in its low nibble and `k + 1` in its
/// high one. The scales follow the bytes, `[N / LANES][K / GROUP][LANES]` f32, and
/// each group's sum is scaled once. `w_ptr` and `s_ptr` are the same buffer, read as
/// bytes and as f32.
///
/// Grid: `(N / LANES, ceil(M / ROW_TILE))`.
#[allow(clippy::multiple_bound_locations)]
#[kernel]
pub fn linear_lanes_i4_forward<T: Triton, D: Float, const LANES: i32, const GROUP: i32>(
    x_ptr: InPtr<T::Pointer<D>>,
    w_ptr: InPtr<T::Pointer<u8>>,
    s_ptr: InPtr<T::Pointer<D>>,
    y_ptr: OutPtr<T::Pointer<D>>,
    M: i32,
    N: i32,
    K: i32,
) where
    T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>,
    T::I32Tensor: Tensor<i32, 1>,
    T::I32Tensor: core::ops::Add<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Sub<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::BitAnd<i32, Output = T::I32Tensor>,
    T::I32Tensor: core::ops::Shr<i32, Output = T::I32Tensor>,
    T::Pointer<D>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<D>>>,
    T::Pointer<u8>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<u8>>>,
{
    let nb = T::program_id(Axis::X);
    let rt = T::program_id(Axis::Y);
    let lanes = T::arange(0, LANES);
    let splat = T::zeros::<i32>(&[LANES]);
    let wbase = nb * (K / 2) * LANES;
    let sbase = N * K / 8 + nb * (K / GROUP) * LANES;
    let r0 = rt * 4;
    let col = nb * LANES;

    if M - r0 == 1 {
        let x0 = r0 * K;
        let mut a0 = T::zeros::<D>(&[LANES]);
        for g in 0..K / GROUP {
            let mut p0 = T::zeros::<D>(&[LANES]);
            let k0 = g * GROUP;
            for j in 0..GROUP / 2 {
                let b = T::cast::<u8, i32>(
                    T::load(
                        w_ptr.add_offsets(lanes + (wbase + (k0 / 2 + j) * LANES)),
                        None,
                        None,
                        &[],
                        None,
                        None,
                        None,
                        false,
                    ),
                    None,
                    false,
                );
                let lo = T::cast::<i32, D>((b & 15) - 8, None, false);
                let hi = T::cast::<i32, D>((b >> 4) - 8, None, false);
                let k = k0 + 2 * j;
                p0 = p0
                    + lo * T::load(
                        x_ptr.add_offsets(splat + (x0 + k)),
                        None,
                        None,
                        &[],
                        None,
                        None,
                        None,
                        false,
                    )
                    + hi * T::load(
                        x_ptr.add_offsets(splat + (x0 + k + 1)),
                        None,
                        None,
                        &[],
                        None,
                        None,
                        None,
                        false,
                    );
            }
            let scale = T::load(
                s_ptr.add_offsets(lanes + (sbase + g * LANES)),
                None,
                None,
                &[],
                None,
                None,
                None,
                false,
            );
            a0 = a0 + p0 * scale;
        }
        T::store(
            y_ptr.add_offsets(lanes + (r0 * N + col)),
            a0,
            None,
            &[],
            None,
            None,
        );
    } else {
        let last = M - 1;
        let r1 = if r0 + 1 < M { r0 + 1 } else { last };
        let r2 = if r0 + 2 < M { r0 + 2 } else { last };
        let r3 = if r0 + 3 < M { r0 + 3 } else { last };
        let x0 = r0 * K;
        let x1 = r1 * K;
        let x2 = r2 * K;
        let x3 = r3 * K;
        let mut a0 = T::zeros::<D>(&[LANES]);
        let mut a1 = T::zeros::<D>(&[LANES]);
        let mut a2 = T::zeros::<D>(&[LANES]);
        let mut a3 = T::zeros::<D>(&[LANES]);
        for g in 0..K / GROUP {
            let mut p0 = T::zeros::<D>(&[LANES]);
            let mut p1 = T::zeros::<D>(&[LANES]);
            let mut p2 = T::zeros::<D>(&[LANES]);
            let mut p3 = T::zeros::<D>(&[LANES]);
            let k0 = g * GROUP;
            for j in 0..GROUP / 2 {
                let b = T::cast::<u8, i32>(
                    T::load(
                        w_ptr.add_offsets(lanes + (wbase + (k0 / 2 + j) * LANES)),
                        None,
                        None,
                        &[],
                        None,
                        None,
                        None,
                        false,
                    ),
                    None,
                    false,
                );
                let lo = T::cast::<i32, D>((b & 15) - 8, None, false);
                let hi = T::cast::<i32, D>((b >> 4) - 8, None, false);
                let k = k0 + 2 * j;
                p0 = p0
                    + lo * T::load(
                        x_ptr.add_offsets(splat + (x0 + k)),
                        None,
                        None,
                        &[],
                        None,
                        None,
                        None,
                        false,
                    )
                    + hi * T::load(
                        x_ptr.add_offsets(splat + (x0 + k + 1)),
                        None,
                        None,
                        &[],
                        None,
                        None,
                        None,
                        false,
                    );
                p1 = p1
                    + lo * T::load(
                        x_ptr.add_offsets(splat + (x1 + k)),
                        None,
                        None,
                        &[],
                        None,
                        None,
                        None,
                        false,
                    )
                    + hi * T::load(
                        x_ptr.add_offsets(splat + (x1 + k + 1)),
                        None,
                        None,
                        &[],
                        None,
                        None,
                        None,
                        false,
                    );
                p2 = p2
                    + lo * T::load(
                        x_ptr.add_offsets(splat + (x2 + k)),
                        None,
                        None,
                        &[],
                        None,
                        None,
                        None,
                        false,
                    )
                    + hi * T::load(
                        x_ptr.add_offsets(splat + (x2 + k + 1)),
                        None,
                        None,
                        &[],
                        None,
                        None,
                        None,
                        false,
                    );
                p3 = p3
                    + lo * T::load(
                        x_ptr.add_offsets(splat + (x3 + k)),
                        None,
                        None,
                        &[],
                        None,
                        None,
                        None,
                        false,
                    )
                    + hi * T::load(
                        x_ptr.add_offsets(splat + (x3 + k + 1)),
                        None,
                        None,
                        &[],
                        None,
                        None,
                        None,
                        false,
                    );
            }
            let scale = T::load(
                s_ptr.add_offsets(lanes + (sbase + g * LANES)),
                None,
                None,
                &[],
                None,
                None,
                None,
                false,
            );
            a0 = a0 + p0 * scale;
            a1 = a1 + p1 * scale;
            a2 = a2 + p2 * scale;
            a3 = a3 + p3 * scale;
        }
        T::store(
            y_ptr.add_offsets(lanes + (r0 * N + col)),
            a0,
            None,
            &[],
            None,
            None,
        );
        T::store(
            y_ptr.add_offsets(lanes + (r1 * N + col)),
            a1,
            None,
            &[],
            None,
            None,
        );
        T::store(
            y_ptr.add_offsets(lanes + (r2 * N + col)),
            a2,
            None,
            &[],
            None,
            None,
        );
        T::store(
            y_ptr.add_offsets(lanes + (r3 * N + col)),
            a3,
            None,
            &[],
            None,
            None,
        );
    }
}

/// Quantises a row-major `[n, k]` matrix to the buffer [`linear_lanes_i4_forward`]
/// reads: symmetric int4 per output row and group of [`I4_GROUP`] inputs
/// (`scale = max |w| / 7`), then the scales.
pub fn quantize_lanes_i4(values: &[f32], n: usize, k: usize) -> Result<Vec<f32>, String> {
    if n % I4_LANES != 0 || k % I4_GROUP != 0 || values.len() != n * k {
        return Err(format!(
            "int4 lanes: {} values are not {n} rows (a multiple of {I4_LANES}) of {k} \
             (a multiple of {I4_GROUP})",
            values.len()
        ));
    }
    let groups = k / I4_GROUP;
    let mut bytes = vec![0u8; n * k / 2];
    let mut scales = vec![0f32; n * groups];
    for (r, row) in values.chunks_exact(k).enumerate() {
        let (block, lane) = (r / I4_LANES, r % I4_LANES);
        for (g, group) in row.chunks_exact(I4_GROUP).enumerate() {
            let max = group.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let scale = if max > 0.0 { max / 7.0 } else { 1.0 };
            scales[(block * groups + g) * I4_LANES + lane] = scale;
            for (i, &v) in group.iter().enumerate() {
                let c = g * I4_GROUP + i;
                let q = ((v / scale).round().clamp(-7.0, 7.0) as i32 + 8) as u8;
                let byte = &mut bytes[(block * (k / 2) + c / 2) * I4_LANES + lane];
                *byte |= if c % 2 == 0 { q } else { q << 4 };
            }
        }
    }
    let mut out: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_bits(u32::from_le_bytes([b[0], b[1], b[2], b[3]])))
        .collect();
    out.extend(scales);
    Ok(out)
}

/// The words [`quantize_lanes_i4`] produces for an `[n, k]` matrix.
pub fn lanes_i4_words(n: usize, k: usize) -> usize {
    n * k / 8 + n * (k / I4_GROUP)
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

/// A bias-free linear on int4 weights, for [`linear_lanes_i4_forward`]. Weight slot
/// `weight_i4`: [`quantize_lanes_i4`]'s buffer, declared `[N, K / 8 + K / I4_GROUP]`
/// words.
pub struct LinearLanesI4Op<D: FloatBytes + Send + Sync + 'static> {
    /// Input features `K` (a multiple of [`I4_GROUP`]).
    pub k: usize,
    /// Output features `N` (a multiple of [`I4_LANES`]).
    pub n: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> LinearLanesI4Op<D> {
    /// Creates the op.
    pub fn new(k: usize, n: usize) -> Self {
        assert!(
            n.is_multiple_of(I4_LANES) && k.is_multiple_of(I4_GROUP),
            "int4 lanes: [{n}, {k}] needs outputs a multiple of {I4_LANES} and inputs of {I4_GROUP}"
        );
        Self {
            k,
            n,
            _phantom: PhantomData,
        }
    }

    /// Whether a `[n, k]` weight fits the kernel.
    pub fn fits(k: usize, n: usize) -> bool {
        n.is_multiple_of(I4_LANES) && k.is_multiple_of(I4_GROUP)
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for LinearLanesI4Op<D> {
    fn name(&self) -> &str {
        "linear_lanes_i4"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[0][0], Some(self.n)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        let kernel = LinearLanesI4Forward::<D>::new(I4_LANES as i32, I4_GROUP as i32);
        Some((
            "linear_lanes_i4_forward".to_string(),
            kernel.source,
            "entry_point".to_string(),
            Arc::new(LinearLanesI4RuntimeOp::<D> {
                k: self.k,
                n: self.n,
                _phantom: PhantomData,
            }),
        ))
    }
}

/// Runtime dispatch for [`LinearLanesI4Op`].
pub struct LinearLanesI4RuntimeOp<D: FloatBytes + Send + Sync + 'static> {
    k: usize,
    n: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> RuntimeOp for LinearLanesI4RuntimeOp<D> {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![self.n, self.k / 8 + self.k / I4_GROUP]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight_i4"]
    }

    fn pack_args(
        &self,
        inputs: &[(RawPtr, &[usize])],
        params: &[RawPtr],
        output: RawPtr,
        output_shape: &[usize],
        _output_row_stride: i32,
        visitor: &mut dyn ArgVisitor,
    ) {
        assert_eq!(inputs[0].1[1], self.k, "int4 lanes: input features");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        visitor.visit_i32(output_shape[0] as i32);
        visitor.visit_i32(self.n as i32);
        visitor.visit_i32(self.k as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        [
            (self.n / I4_LANES) as u32,
            output_shape[0].div_ceil(ROW_TILE) as u32,
            1,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_lanes_i4_places_nibbles_and_scales() {
        let (n, k) = (I4_LANES, I4_GROUP * 2);
        // Row r, input c: r - 16 in group 0, (r - 16) * 2 in group 1.
        let values: Vec<f32> = (0..n * k)
            .map(|i| {
                let (r, c) = (i / k, i % k);
                let v = (r as f32 - 16.0) / 16.0 * 7.0;
                if c < I4_GROUP { v } else { v * 2.0 }
            })
            .collect();
        let buf = quantize_lanes_i4(&values, n, k).unwrap();
        assert_eq!(buf.len(), lanes_i4_words(n, k));
        let bytes: Vec<u8> = buf[..n * k / 8]
            .iter()
            .flat_map(|w| w.to_bits().to_le_bytes())
            .collect();
        let scales = &buf[n * k / 8..];
        for r in [0usize, 5, 16, 31] {
            for c in [0usize, 1, I4_GROUP - 1, I4_GROUP, k - 1] {
                let byte = bytes[(c / 2) * I4_LANES + r];
                let q = if c % 2 == 0 { byte & 15 } else { byte >> 4 } as i32 - 8;
                let g = c / I4_GROUP;
                let got = q as f32 * scales[g * I4_LANES + r];
                assert!((got - values[r * k + c]).abs() <= scales[g * I4_LANES + r] / 2.0 + 1e-6);
            }
        }
    }

    #[test]
    fn test_quantize_lanes_i4_refuses_a_ragged_shape() {
        assert!(quantize_lanes_i4(&[0.0; 32 * 48], 32, 48).is_err());
    }
}
