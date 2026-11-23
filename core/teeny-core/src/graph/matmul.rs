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

use alloc::vec::Vec;

/// How a `MatMul` maps onto the batched GEMM kernel.
///
/// The kernel views each operand as a 2-D row-major matrix and selects a batch by
/// skipping rows: the left operand is `[batch * m, k]`, the right `[batch * k, n]`
/// (or `[k, n]` when it is shared), and the output `[batch * m, n]`. That is exactly
/// how ONNX lays a batched `MatMul` out, so no copy is needed to get there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatMulPlan {
    /// How many independent `[m, k] @ [k, n]` products to compute.
    pub batch: usize,
    /// Rows of the left operand and of the output.
    pub m: usize,
    /// Columns of the right operand and of the output.
    pub n: usize,
    /// The contracted extent.
    pub k: usize,
    /// Rows to skip in the left operand per batch: `m`, or 0 when it is shared.
    pub a_batch_rows: usize,
    /// Rows to skip in the right operand per batch: `k`, or 0 when it is shared.
    pub b_batch_rows: usize,
}

/// Why a `MatMul`'s shapes do not map onto the batched GEMM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatMulError {
    /// A rank-1 or rank-0 operand. ONNX promotes a vector operand to a matrix and
    /// drops the axis again afterwards; this plan does not, because the kernel would
    /// then need to write a differently ranked output than the one it was given.
    OperandRankTooLow {
        /// Which operand, 0 for the left.
        operand: usize,
        /// Its rank.
        rank: usize,
    },
    /// The contracted extents disagree.
    ContractionMismatch {
        /// The left operand's last extent.
        a: usize,
        /// The right operand's second-to-last extent.
        b: usize,
    },
    /// The batch prefixes do not broadcast against each other.
    BatchBroadcast {
        /// The left operand's batch prefix.
        a: Vec<usize>,
        /// The right operand's batch prefix.
        b: Vec<usize>,
    },
    /// A batch prefix broadcasts, but to something the kernel's single row stride
    /// cannot address: the operand neither carries the whole batch nor is shared
    /// across all of it (`[2, 1, m, k]` against `[1, 3, k, n]`, say). Reaching the
    /// elements would need one stride per batch axis.
    BatchNotOneStride {
        /// Which operand, 0 for the left.
        operand: usize,
        /// That operand's batch prefix.
        prefix: Vec<usize>,
        /// The broadcast batch shape it would have to address.
        batch: Vec<usize>,
    },
}

impl core::fmt::Display for MatMulError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OperandRankTooLow { operand, rank } => write!(
                f,
                "operand {operand} has rank {rank}; a vector operand is not supported, \
                 give it an explicit length-1 axis"
            ),
            Self::ContractionMismatch { a, b } => write!(
                f,
                "the left operand contracts over {a} and the right over {b}"
            ),
            Self::BatchBroadcast { a, b } => write!(
                f,
                "batch prefixes {a:?} and {b:?} do not broadcast against each other"
            ),
            Self::BatchNotOneStride {
                operand,
                prefix,
                batch,
            } => write!(
                f,
                "operand {operand}'s batch prefix {prefix:?} is neither the whole \
                 batch {batch:?} nor shared across it, so one row stride cannot \
                 address it"
            ),
        }
    }
}

/// Plan a `MatMul` of `a_shape` by `b_shape` as a batched GEMM.
///
/// The batch axes are the shapes' prefixes, broadcast against each other the way
/// ONNX does. An operand that carries the whole batch is indexed by a row stride; one
/// whose prefix is all ones is shared and gets a stride of 0. Any other broadcast is
/// refused by name rather than approximated.
///
/// When the right operand is shared, the batch is folded into `m`: the left operand's
/// rows are then contiguous across the whole batch, and so are the output's, which
/// makes the product one larger 2-D GEMM. That is worth doing because the batch axes
/// of a weight multiply are often long and `m` short, and a batch axis costs a whole
/// tile row where `m` can fill one.
pub fn matmul_plan(a_shape: &[usize], b_shape: &[usize]) -> Result<MatMulPlan, MatMulError> {
    for (operand, shape) in [a_shape, b_shape].iter().enumerate() {
        if shape.len() < 2 {
            return Err(MatMulError::OperandRankTooLow {
                operand,
                rank: shape.len(),
            });
        }
    }

    let m = a_shape[a_shape.len() - 2];
    let k = a_shape[a_shape.len() - 1];
    let b_k = b_shape[b_shape.len() - 2];
    let n = b_shape[b_shape.len() - 1];
    if k != b_k {
        return Err(MatMulError::ContractionMismatch { a: k, b: b_k });
    }

    let a_prefix = &a_shape[..a_shape.len() - 2];
    let b_prefix = &b_shape[..b_shape.len() - 2];

    // Right-aligned broadcast of the two prefixes, as ONNX specifies.
    let rank = a_prefix.len().max(b_prefix.len());
    let mut batch_shape = Vec::with_capacity(rank);
    for i in 0..rank {
        let at = |p: &[usize]| -> usize {
            let offset = rank - p.len();
            if i < offset { 1 } else { p[i - offset] }
        };
        let (x, y) = (at(a_prefix), at(b_prefix));
        let extent = match (x, y) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => {
                return Err(MatMulError::BatchBroadcast {
                    a: a_prefix.to_vec(),
                    b: b_prefix.to_vec(),
                });
            }
        };
        batch_shape.push(extent);
    }
    let batch: usize = batch_shape.iter().product();

    // One row stride per operand can address the batch only if the operand either
    // carries all of it or none of it.
    let rows = |operand: usize, prefix: &[usize], per_batch: usize| -> Result<usize, MatMulError> {
        let count: usize = prefix.iter().product();
        // Shared comes first so that an unbatched product, where both are 1, reports
        // no stride rather than a stride nothing will ever multiply.
        if count == 1 {
            Ok(0)
        } else if count == batch {
            Ok(per_batch)
        } else {
            Err(MatMulError::BatchNotOneStride {
                operand,
                prefix: prefix.to_vec(),
                batch: batch_shape.clone(),
            })
        }
    };
    let a_batch_rows = rows(0, a_prefix, m)?;
    let b_batch_rows = rows(1, b_prefix, k)?;

    // A shared right operand means the left operand's and the output's rows run
    // contiguously across the batch, so the whole thing is one taller GEMM.
    if batch > 1 && b_batch_rows == 0 && a_batch_rows == m {
        return Ok(MatMulPlan {
            batch: 1,
            m: m * batch,
            n,
            k,
            a_batch_rows: 0,
            b_batch_rows: 0,
        });
    }

    Ok(MatMulPlan {
        batch,
        m,
        n,
        k,
        a_batch_rows,
        b_batch_rows,
    })
}

#[cfg(test)]
mod tests {
    use super::{MatMulError, MatMulPlan, matmul_plan};
    use alloc::string::ToString;

    fn plan(a: &[usize], b: &[usize]) -> MatMulPlan {
        matmul_plan(a, b).expect("these shapes should plan")
    }

    #[test]
    fn test_a_plain_matrix_product_has_no_batch() {
        let p = plan(&[128, 64], &[64, 32]);
        assert_eq!(p.batch, 1);
        assert_eq!((p.m, p.k, p.n), (128, 64, 32));
        assert_eq!((p.a_batch_rows, p.b_batch_rows), (0, 0));
    }

    #[test]
    fn test_a_shared_weight_folds_the_batch_into_the_rows() {
        // [2, 1921, 1024] @ [1024, 1024]: the left operand's rows run contiguously
        // across the batch, so this is one 3842-row GEMM rather than two.
        let p = plan(&[2, 1921, 1024], &[1024, 1024]);
        assert_eq!(p.batch, 1);
        assert_eq!((p.m, p.k, p.n), (2 * 1921, 1024, 1024));

        // The same holds however many batch axes there are.
        let p = plan(&[1, 120, 160, 128], &[128, 512]);
        assert_eq!(p.batch, 1);
        assert_eq!((p.m, p.k, p.n), (120 * 160, 128, 512));
    }

    #[test]
    fn test_two_batched_operands_keep_their_batch() {
        // Attention: [2, 16, 1921, 64] @ [2, 16, 64, 1921], 32 independent products.
        let p = plan(&[2, 16, 1921, 64], &[2, 16, 64, 1921]);
        assert_eq!(p.batch, 32);
        assert_eq!((p.m, p.k, p.n), (1921, 64, 1921));
        assert_eq!(p.a_batch_rows, 1921);
        assert_eq!(p.b_batch_rows, 64);
    }

    #[test]
    fn test_a_shared_left_operand_gets_a_zero_stride() {
        let p = plan(&[1, 4, 8], &[3, 8, 5]);
        assert_eq!(p.batch, 3);
        assert_eq!((p.m, p.k, p.n), (4, 8, 5));
        assert_eq!(p.a_batch_rows, 0);
        assert_eq!(p.b_batch_rows, 8);
    }

    #[test]
    fn test_a_prefix_shorter_than_the_other_broadcasts_from_the_right() {
        let p = plan(&[4, 8], &[3, 8, 5]);
        assert_eq!(p.batch, 3);
        assert_eq!(p.a_batch_rows, 0);
    }

    #[test]
    fn test_a_vector_operand_is_refused_rather_than_promoted() {
        assert_eq!(
            matmul_plan(&[3, 4], &[4]),
            Err(MatMulError::OperandRankTooLow {
                operand: 1,
                rank: 1
            })
        );
    }

    #[test]
    fn test_the_contracted_extents_must_agree() {
        assert_eq!(
            matmul_plan(&[3, 4], &[5, 2]),
            Err(MatMulError::ContractionMismatch { a: 4, b: 5 })
        );
    }

    #[test]
    fn test_a_prefix_that_does_not_broadcast_is_refused() {
        assert!(matches!(
            matmul_plan(&[2, 3, 4], &[3, 4, 5]),
            Err(MatMulError::BatchBroadcast { .. })
        ));
    }

    #[test]
    fn test_a_batch_that_one_row_stride_cannot_address_is_refused() {
        // [2, 1] against [1, 3] broadcasts to a batch of 6, but neither operand holds
        // all six nor just one: reaching an element needs a stride per batch axis.
        let err = matmul_plan(&[2, 1, 4, 8], &[1, 3, 8, 5]).expect_err("not one stride");
        assert!(matches!(
            err,
            MatMulError::BatchNotOneStride { operand: 0, .. }
        ));
        assert!(err.to_string().contains("[2, 1]"), "{err}");
    }
}
