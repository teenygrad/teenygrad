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

//! Historical hand-written fused Conv2d+BN+SiLU kernels.
//!
//! Not used by Anduin (teenygrad-1bf.8 hand-kernel path removed). Kept for
//! microbenchmarks only — do not wire these back into graph lowering.

pub mod conv2d_bn_silu;
pub mod conv2d_bn_silu_gemm;
pub mod conv2d_bn_silu_tiled;

/// Prefold BatchNorm2d inference affine into scale/shift (bench / tooling helper).
///
/// ```text
/// bn_scale[c] = gamma[c] / sqrt(var[c] + eps)
/// bn_shift[c] = beta[c]  - bn_scale[c] * mean[c]
/// ```
pub fn prefold_bn_affine(
    gamma: &[f32],
    beta: &[f32],
    mean: &[f32],
    var: &[f32],
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = gamma.len();
    debug_assert_eq!(beta.len(), n);
    debug_assert_eq!(mean.len(), n);
    debug_assert_eq!(var.len(), n);
    let mut scale = Vec::with_capacity(n);
    let mut shift = Vec::with_capacity(n);
    for i in 0..n {
        let s = gamma[i] / (var[i] + eps).sqrt();
        scale.push(s);
        shift.push(beta[i] - s * mean[i]);
    }
    (scale, shift)
}

#[cfg(test)]
mod tests {
    use super::prefold_bn_affine;

    #[test]
    fn prefold_identity_bn() {
        let gamma = vec![1.0f32];
        let beta = vec![0.0f32];
        let mean = vec![0.0f32];
        let var = vec![1.0f32];
        let (scale, shift) = prefold_bn_affine(&gamma, &beta, &mean, &var, 0.0);
        assert!((scale[0] - 1.0).abs() < 1e-6);
        assert!(shift[0].abs() < 1e-6);
    }
}
