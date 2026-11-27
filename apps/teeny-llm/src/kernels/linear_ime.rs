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

//! A linear on the SpacemiT K3 A100 cores' int8 matrix engine: `y = x W^T`, with
//! int8 weights (one scale per output row) and int8 activations (one scale per token).
//!
//! Decode reads every linear weight once a token, so its speed is the weight bytes
//! over the memory bandwidth the kernel reaches. The A100 cores read faster than the
//! X100 ones, but run ordinary vector code about four times slower: only through
//! their matrix instructions are they fast. Those the kernel compiler cannot emit, so
//! both kernels here are native (`native/linear_ime.c`, `native/quant_pack_i8.c`):
//!
//! - [`QuantPackI8Op`] (X100) quantises `x` per token into the matrix engine's
//!   8-token tiles.
//! - [`LinearImeOp`] (A100) runs one program per 8 outputs over every tile. Its
//!   weights are [`quantize_ime_i8`]'s, packed at load.

use core::marker::PhantomData;
use std::any::Any;
use std::sync::Arc;
use teeny_core::device::program::ArgVisitor;
use teeny_core::dtype::FloatBytes;
use teeny_core::graph::{CustomOp, Shape};
use teeny_core::model::{RawPtr, RuntimeOp};

/// Outputs and tokens per matrix-engine tile.
pub const TILE: usize = 8;

/// Elements of `K` per tile row.
pub const CHUNK: usize = 16;

/// `K` must be a multiple of this: the kernel keeps four chunks in flight.
pub const K_MULTIPLE: usize = 4 * CHUNK;

/// Tokens rounded up to whole tiles.
fn padded(m: usize) -> usize {
    m.div_ceil(TILE) * TILE
}

/// Quantises a row-major `[n, k]` matrix to the buffer [`LinearImeOp`] reads:
/// symmetric int8 per output row (`scale = max |w| / 127`), in
/// `[n / 8][k / 16][8][16]` order, four bytes to a word, then the `n` scales.
pub fn quantize_ime_i8(values: &[f32], n: usize, k: usize) -> Result<Vec<f32>, String> {
    if n % TILE != 0 || k % K_MULTIPLE != 0 || values.len() != n * k {
        return Err(format!(
            "ime int8: {} values are not {n} rows (a multiple of {TILE}) of {k} (a multiple of {K_MULTIPLE})",
            values.len()
        ));
    }
    let chunks = k / CHUNK;
    let mut bytes = vec![0u8; n * k];
    let mut scales = Vec::with_capacity(n);
    for (r, row) in values.chunks_exact(k).enumerate() {
        let max = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = if max > 0.0 { max / 127.0 } else { 1.0 };
        let (block, lane) = (r / TILE, r % TILE);
        for (col, &v) in row.iter().enumerate() {
            let q = (v / scale).round().clamp(-127.0, 127.0) as i8;
            let (c, i) = (col / CHUNK, col % CHUNK);
            bytes[((block * chunks + c) * TILE + lane) * CHUNK + i] = q as u8;
        }
        scales.push(if max > 0.0 { scale } else { 0.0 });
    }
    let mut out: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_bits(u32::from_le_bytes([b[0], b[1], b[2], b[3]])))
        .collect();
    out.extend(scales);
    Ok(out)
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

/// `x` of `[M, K]` to `[M_pad, K / 4 + 1]` words: the int8 tiles, then one scale
/// per padded token (`M_pad` is `M` rounded up to 8). Rows past `M` hold whatever
/// was in the buffer; the linear never stores the outputs they produce.
pub struct QuantPackI8Op<D: FloatBytes + Send + Sync + 'static> {
    /// `K` (a multiple of [`K_MULTIPLE`]).
    pub k: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> QuantPackI8Op<D> {
    /// Creates the op.
    pub fn new(k: usize) -> Self {
        assert!(k.is_multiple_of(K_MULTIPLE), "quant pack: {k} is not a multiple of {K_MULTIPLE}");
        Self { k, _phantom: PhantomData }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for QuantPackI8Op<D> {
    fn name(&self) -> &str {
        "quant_pack_i8"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[0][0].map(padded), Some(self.k / 4 + 1)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        Some((
            "quant_pack_i8".to_string(),
            "// native: native/quant_pack_i8.c".to_string(),
            "quant_pack_i8_entry_point".to_string(),
            Arc::new(QuantPackI8RuntimeOp { k: self.k }),
        ))
    }
}

/// Runtime dispatch for [`QuantPackI8Op`].
pub struct QuantPackI8RuntimeOp {
    k: usize,
}

impl RuntimeOp for QuantPackI8RuntimeOp {
    fn n_activation_inputs(&self) -> usize {
        1
    }

    fn native_library(&self) -> Option<std::path::PathBuf> {
        Some(super::native::library("quant_pack_i8").unwrap_or_else(|e| panic!("{e:#}")))
    }

    fn compute_concrete_output_shape(&self, input_shapes: &[&[usize]], _resolved: &[usize]) -> Vec<usize> {
        vec![padded(input_shapes[0][0]), self.k / 4 + 1]
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
        assert_eq!(inputs[0].1[1], self.k, "quant pack: input width");
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(output);
        visitor.visit_i32(self.k as i32);
        visitor.visit_i32(output_shape[0] as i32);
        visitor.visit_i32(inputs[0].1[0] as i32);
    }

    fn grid(&self, output_shape: &[usize]) -> [u32; 3] {
        // One program per padded row; those past the real tokens return at once.
        [output_shape[0] as u32, 1, 1]
    }
}

// ---------------------------------------------------------------------------
// The linear
// ---------------------------------------------------------------------------

/// `y = x W^T` on the A100 matrix engine. Inputs: [`QuantPackI8Op`]'s output, and
/// `x` itself, whose row count is the real token count. Weight slot `weight_ime`:
/// [`quantize_ime_i8`]'s buffer, declared `[N, K / 4 + 1]` words.
pub struct LinearImeOp<D: FloatBytes + Send + Sync + 'static> {
    /// `K`.
    pub k: usize,
    /// `N` (a multiple of [`TILE`]).
    pub n: usize,
    _phantom: PhantomData<D>,
}

impl<D: FloatBytes + Send + Sync + 'static> LinearImeOp<D> {
    /// Creates the op.
    pub fn new(k: usize, n: usize) -> Self {
        assert!(n.is_multiple_of(TILE), "linear ime: {n} outputs are not a multiple of {TILE}");
        assert!(k.is_multiple_of(K_MULTIPLE), "linear ime: {k} is not a multiple of {K_MULTIPLE}");
        Self { k, n, _phantom: PhantomData }
    }
}

impl<D: FloatBytes + Send + Sync + 'static> CustomOp for LinearImeOp<D> {
    fn name(&self) -> &str {
        "linear_ime"
    }

    fn infer_output_shape(&self, input_shapes: &[&Shape]) -> Shape {
        vec![input_shapes[1][0], Some(self.n)]
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn lower(&self) -> Option<(String, String, String, Arc<dyn RuntimeOp>)> {
        Some((
            "linear_ime".to_string(),
            "// native: native/linear_ime.c".to_string(),
            "linear_ime_a100_entry_point".to_string(),
            Arc::new(LinearImeRuntimeOp { k: self.k, n: self.n }),
        ))
    }
}

/// Runtime dispatch for [`LinearImeOp`].
pub struct LinearImeRuntimeOp {
    k: usize,
    n: usize,
}

impl RuntimeOp for LinearImeRuntimeOp {
    fn n_activation_inputs(&self) -> usize {
        2
    }

    fn native_library(&self) -> Option<std::path::PathBuf> {
        Some(super::native::library("linear_ime").unwrap_or_else(|e| panic!("{e:#}")))
    }

    fn compute_concrete_output_shape(&self, input_shapes: &[&[usize]], _resolved: &[usize]) -> Vec<usize> {
        vec![input_shapes[1][0], self.n]
    }

    fn param_shapes(&self, _input_shapes: &[&[usize]], _output_shape: &[usize]) -> Vec<Vec<usize>> {
        vec![vec![self.n, self.k / 4 + 1]]
    }

    fn param_names(&self) -> &'static [&'static str] {
        &["weight_ime"]
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
        let m_pad = inputs[0].1[0];
        assert_eq!(inputs[0].1[1], self.k / 4 + 1, "linear ime: packed activation width");
        assert_eq!(inputs[1].1[1], self.k, "linear ime: input width");
        assert!(m_pad >= output_shape[0], "linear ime: {m_pad} packed rows for {} tokens", output_shape[0]);
        visitor.visit_ptr(inputs[0].0);
        visitor.visit_ptr(params[0]);
        visitor.visit_ptr(output);
        visitor.visit_i32(output_shape[0] as i32);
        visitor.visit_i32(self.n as i32);
        visitor.visit_i32(self.k as i32);
        visitor.visit_i32(m_pad as i32);
    }

    fn grid(&self, _output_shape: &[usize]) -> [u32; 3] {
        // Four row blocks per program, as `BLOCKS` in native/linear_ime.c.
        [(self.n / TILE).div_ceil(4) as u32, 1, 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each byte lands where the kernel reads it: output row r's chunk c is 16 bytes
    /// at ((r / 8 * chunks + c) * 8 + r % 8) * 16.
    #[test]
    fn packing_puts_each_row_chunk_in_its_tile_slot() {
        let (n, k) = (16usize, 64usize);
        let w: Vec<f32> = (0..n * k).map(|i| ((i % 251) as f32 - 125.0) / 125.0).collect();
        let buf = quantize_ime_i8(&w, n, k).unwrap();
        let bytes: Vec<u8> = buf[..n * k / 4].iter().flat_map(|f| f.to_bits().to_le_bytes()).collect();
        let scales = &buf[n * k / 4..];
        let chunks = k / CHUNK;
        for r in 0..n {
            for col in 0..k {
                let q = bytes[((r / 8 * chunks + col / 16) * 8 + r % 8) * 16 + col % 16] as i8;
                let want = (w[r * k + col] / scales[r]).round() as i8;
                assert_eq!(q, want, "row {r} col {col}");
            }
        }
    }
}
