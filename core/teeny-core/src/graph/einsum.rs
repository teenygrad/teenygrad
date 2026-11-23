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

use alloc::{string::String, vec, vec::Vec};

/// The ranks an [`EinsumPlan`] can describe.
///
/// Four is what keeps the kernel inside the 20-argument limit its ABI imposes: three
/// pointers and sixteen scalars, of which twelve are the three length-4 stride and
/// extent vectors. Attention and bilinear contractions, the forms that turn up in
/// exports, are rank 3 and 4.
pub const EINSUM_MAX_RANK: usize = 4;

/// How a two-operand `Einsum` maps onto a strided contraction.
///
/// Every output element is
///
/// ```text
/// y[coords] = sum over c in 0..k of
///     a[dot(coords, a_strides) + c * a_contract_stride]
///   * b[dot(coords, b_strides) + c * b_contract_stride]
/// ```
///
/// A label that an operand does not mention gets stride 0 there, which is how a label
/// free on one side is broadcast along the other's axes without any special case. A
/// label in both operands and the output is a batch axis and simply has a stride on
/// both sides.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EinsumPlan {
    /// The output's extents, in the order the equation's right-hand side gives.
    pub out_shape: Vec<usize>,
    /// Stride into the left operand per output coordinate; 0 where it does not apply.
    pub a_strides: Vec<i64>,
    /// Stride into the right operand per output coordinate.
    pub b_strides: Vec<i64>,
    /// The contracted label's extent.
    pub k: usize,
    /// Stride into the left operand along the contracted label.
    pub a_contract_stride: i64,
    /// Stride into the right operand along the contracted label.
    pub b_contract_stride: i64,
}

/// Why an `Einsum` equation cannot be planned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EinsumError {
    /// The equation is not `<labels>,<labels>-><labels>`.
    NotTwoOperandForm,
    /// An operand's label count does not match its tensor's rank.
    RankMismatch {
        /// Which operand, 0 or 1.
        operand: usize,
        /// Labels the equation gives it.
        labels: usize,
        /// Dimensions the tensor has.
        rank: usize,
    },
    /// A label appears twice in one operand, which takes a diagonal.
    RepeatedLabel {
        /// The offending label.
        label: char,
    },
    /// A label appears twice in the output, which ONNX and NumPy both reject: each
    /// output axis must be a distinct label.
    RepeatedOutputLabel {
        /// The offending label.
        label: char,
    },
    /// A character that is not an ASCII letter where a label is expected.
    InvalidLabel {
        /// The offending character.
        label: char,
    },
    /// A label's extent differs between the operands.
    ExtentMismatch {
        /// The offending label.
        label: char,
        /// Its extent in the left operand.
        a: usize,
        /// Its extent in the right operand.
        b: usize,
    },
    /// The output names a label neither operand provides.
    UnknownOutputLabel {
        /// The offending label.
        label: char,
    },
    /// A label appears in one operand only and not in the output, which asks for that
    /// operand to be summed along it before the contraction.
    PartialReduction {
        /// The offending label.
        label: char,
    },
    /// Not exactly one contracted label.
    ///
    /// None means the equation is elementwise or an outer product rather than a
    /// contraction; more than one needs a nested sum the kernel does not do.
    ContractedLabels {
        /// How many labels are summed over.
        count: usize,
    },
    /// A rank beyond [`EINSUM_MAX_RANK`].
    RankTooHigh {
        /// The rank asked for.
        rank: usize,
    },
    /// An ellipsis, which stands for an unknown number of batch axes.
    Ellipsis,
}

impl core::fmt::Display for EinsumError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EinsumError::NotTwoOperandForm => write!(
                f,
                "only the two-operand form 'ij,jk->ik' is supported, with an explicit \
                 output"
            ),
            EinsumError::RankMismatch {
                operand,
                labels,
                rank,
            } => write!(f, "operand {operand} has {labels} labels but rank {rank}"),
            EinsumError::RepeatedLabel { label } => write!(
                f,
                "label '{label}' appears twice in one operand, which takes a diagonal"
            ),
            EinsumError::RepeatedOutputLabel { label } => write!(
                f,
                "label '{label}' appears twice in the output; each output axis must be a \
                 distinct label"
            ),
            EinsumError::InvalidLabel { label } => write!(
                f,
                "'{label}' is not a valid label; labels are ASCII letters, one per axis"
            ),
            EinsumError::ExtentMismatch { label, a, b } => write!(
                f,
                "label '{label}' is {a} in the left operand and {b} in the right"
            ),
            EinsumError::UnknownOutputLabel { label } => {
                write!(f, "the output names '{label}', which neither operand has")
            }
            EinsumError::PartialReduction { label } => write!(
                f,
                "label '{label}' is in one operand only and not in the output, so it \
                 asks for a reduction before the contraction"
            ),
            EinsumError::ContractedLabels { count } => write!(
                f,
                "{count} contracted labels; exactly one is supported, and {}",
                if *count == 0 {
                    "none means this is elementwise or an outer product rather than a \
                     contraction"
                } else {
                    "more than one needs a nested sum"
                }
            ),
            EinsumError::RankTooHigh { rank } => write!(
                f,
                "rank {rank} exceeds the {EINSUM_MAX_RANK} this contraction indexes"
            ),
            EinsumError::Ellipsis => write!(
                f,
                "an ellipsis stands for an unknown number of batch axes, which cannot \
                 be planned statically"
            ),
        }
    }
}

/// Plans a two-operand `Einsum` against its operands' shapes.
///
/// The equation is ONNX's: lowercase or uppercase letters, one per axis, operands
/// separated by a comma and the output after `->`. Whitespace is ignored.
pub fn einsum_plan(
    equation: &str,
    a_shape: &[usize],
    b_shape: &[usize],
) -> Result<EinsumPlan, EinsumError> {
    let compact: String = equation.chars().filter(|c| !c.is_whitespace()).collect();
    // After removing whitespace, so that a spaced-out `. . .` is still recognised as an
    // ellipsis rather than reported as three invalid labels.
    if compact.contains("...") {
        return Err(EinsumError::Ellipsis);
    }
    let (inputs, output) = compact
        .split_once("->")
        .ok_or(EinsumError::NotTwoOperandForm)?;
    let (a_labels, b_labels) = inputs
        .split_once(',')
        .ok_or(EinsumError::NotTwoOperandForm)?;
    if b_labels.contains(',') || output.contains("->") {
        return Err(EinsumError::NotTwoOperandForm);
    }

    let a_labels: Vec<char> = a_labels.chars().collect();
    let b_labels: Vec<char> = b_labels.chars().collect();
    let out_labels: Vec<char> = output.chars().collect();

    if let Some(&label) = a_labels
        .iter()
        .chain(&b_labels)
        .chain(&out_labels)
        .find(|c| !c.is_ascii_alphabetic())
    {
        return Err(EinsumError::InvalidLabel { label });
    }

    for (operand, (labels, shape)) in [(&a_labels, a_shape), (&b_labels, b_shape)]
        .into_iter()
        .enumerate()
    {
        if labels.len() != shape.len() {
            return Err(EinsumError::RankMismatch {
                operand,
                labels: labels.len(),
                rank: shape.len(),
            });
        }
        if shape.len() > EINSUM_MAX_RANK {
            return Err(EinsumError::RankTooHigh { rank: shape.len() });
        }
        // A repeated label within one operand asks for its diagonal, which is a
        // different operation from a contraction and is not expressible as a stride.
        for (i, &label) in labels.iter().enumerate() {
            if labels[..i].contains(&label) {
                return Err(EinsumError::RepeatedLabel { label });
            }
        }
    }
    if out_labels.len() > EINSUM_MAX_RANK {
        return Err(EinsumError::RankTooHigh {
            rank: out_labels.len(),
        });
    }
    // A repeated output label would read the operands along a diagonal and write an
    // output of the wrong rank; it is not a valid equation, so refuse it by name.
    for (i, &label) in out_labels.iter().enumerate() {
        if out_labels[..i].contains(&label) {
            return Err(EinsumError::RepeatedOutputLabel { label });
        }
    }

    // Each label's extent and where it sits in each operand.
    let axis_of = |labels: &[char], label: char| labels.iter().position(|&l| l == label);
    let extent = |label: char| -> Option<usize> {
        axis_of(&a_labels, label)
            .map(|i| a_shape[i])
            .or_else(|| axis_of(&b_labels, label).map(|i| b_shape[i]))
    };
    for &label in a_labels.iter() {
        if let (Some(i), Some(j)) = (axis_of(&a_labels, label), axis_of(&b_labels, label))
            && a_shape[i] != b_shape[j]
        {
            return Err(EinsumError::ExtentMismatch {
                label,
                a: a_shape[i],
                b: b_shape[j],
            });
        }
    }

    // Contracted labels are those in both operands but not in the output. A label in
    // one operand only and not in the output would need that operand summed first,
    // which this plan cannot express, so it is an error rather than a silent sum.
    let mut contracted: Vec<char> = Vec::new();
    for &label in a_labels.iter().chain(b_labels.iter()) {
        if out_labels.contains(&label) || contracted.contains(&label) {
            continue;
        }
        let in_a = axis_of(&a_labels, label).is_some();
        let in_b = axis_of(&b_labels, label).is_some();
        if in_a && in_b {
            contracted.push(label);
        } else {
            return Err(EinsumError::PartialReduction { label });
        }
    }
    if contracted.len() != 1 {
        return Err(EinsumError::ContractedLabels {
            count: contracted.len(),
        });
    }
    let c = contracted[0];

    let a_src = contiguous_strides(a_shape);
    let b_src = contiguous_strides(b_shape);

    let mut out_shape = Vec::with_capacity(out_labels.len());
    let mut a_strides = Vec::with_capacity(out_labels.len());
    let mut b_strides = Vec::with_capacity(out_labels.len());
    for &label in &out_labels {
        let Some(e) = extent(label) else {
            return Err(EinsumError::UnknownOutputLabel { label });
        };
        out_shape.push(e);
        // Stride 0 where an operand does not mention the label: every coordinate along
        // it then reads the same element, which is exactly the broadcast a free label
        // needs.
        a_strides.push(axis_of(&a_labels, label).map_or(0, |i| a_src[i]));
        b_strides.push(axis_of(&b_labels, label).map_or(0, |i| b_src[i]));
    }

    Ok(EinsumPlan {
        out_shape,
        a_strides,
        b_strides,
        k: extent(c).unwrap_or(0),
        a_contract_stride: axis_of(&a_labels, c).map_or(0, |i| a_src[i]),
        b_contract_stride: axis_of(&b_labels, c).map_or(0, |i| b_src[i]),
    })
}

/// Row-major strides for `shape`, in elements.
fn contiguous_strides(shape: &[usize]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1] as i64;
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{string::ToString, vec};

    /// The equation the FoundationStereo export uses: a batch pair, one free label on
    /// each side and one contracted label.
    #[test]
    fn test_plans_a_batched_outer_contraction() {
        let plan = einsum_plan("aijk,aijh->ajkh", &[1, 224, 120, 160], &[1, 224, 120, 160])
            .expect("should plan");
        assert_eq!(plan.out_shape, vec![1, 120, 160, 160]);
        assert_eq!(plan.k, 224);

        // a is axis 0 of both, j is axis 2 of both, k is axis 3 of A only and h axis 3
        // of B only, so each gets a stride on one side and 0 on the other.
        let a = contiguous_strides(&[1, 224, 120, 160]);
        assert_eq!(plan.a_strides, vec![a[0], a[2], a[3], 0]);
        assert_eq!(plan.b_strides, vec![a[0], a[2], 0, a[3]]);
        // The contracted label i is axis 1 of both.
        assert_eq!(plan.a_contract_stride, a[1]);
        assert_eq!(plan.b_contract_stride, a[1]);
    }

    /// A batched matrix multiply is one instance of the same plan, which is the point
    /// of planning rather than special-casing.
    #[test]
    fn test_plans_a_batched_matmul() {
        let plan = einsum_plan("bij,bjk->bik", &[4, 2, 3], &[4, 3, 5]).expect("should plan");
        assert_eq!(plan.out_shape, vec![4, 2, 5]);
        assert_eq!(plan.k, 3);
        // b and i index A; k does not, so its stride there is 0.
        assert_eq!(plan.a_strides, vec![6, 3, 0]);
        // b and k index B; i does not.
        assert_eq!(plan.b_strides, vec![15, 0, 1]);
        assert_eq!(plan.a_contract_stride, 1);
        assert_eq!(plan.b_contract_stride, 5);
    }

    /// Plain 2-D matmul, and the transposed forms, which differ only in the strides.
    #[test]
    fn test_plans_the_plain_and_transposed_matmuls() {
        let plan = einsum_plan("ij,jk->ik", &[2, 3], &[3, 4]).expect("ij,jk");
        assert_eq!(plan.out_shape, vec![2, 4]);
        assert_eq!((plan.a_contract_stride, plan.b_contract_stride), (1, 4));

        // Contracting A's first axis instead: the stride along the sum changes, not
        // the shape of the plan.
        let plan = einsum_plan("ji,jk->ik", &[3, 2], &[3, 4]).expect("ji,jk");
        assert_eq!(plan.out_shape, vec![2, 4]);
        assert_eq!((plan.a_contract_stride, plan.b_contract_stride), (2, 4));
    }

    /// The output may name its labels in any order, and the strides follow it.
    #[test]
    fn test_the_output_order_drives_the_strides() {
        let forward = einsum_plan("ij,jk->ik", &[2, 3], &[3, 4]).expect("ik");
        let swapped = einsum_plan("ij,jk->ki", &[2, 3], &[3, 4]).expect("ki");
        assert_eq!(swapped.out_shape, vec![4, 2]);
        // Same mapping, read in the other order.
        assert_eq!(
            swapped.a_strides,
            vec![forward.a_strides[1], forward.a_strides[0]]
        );
        assert_eq!(
            swapped.b_strides,
            vec![forward.b_strides[1], forward.b_strides[0]]
        );
    }

    #[test]
    fn test_rejects_what_it_cannot_express() {
        // No contraction at all: elementwise or an outer product.
        assert_eq!(
            einsum_plan("ij,ij->ij", &[2, 3], &[2, 3]),
            Err(EinsumError::ContractedLabels { count: 0 })
        );
        assert_eq!(
            einsum_plan("i,j->ij", &[2], &[3]),
            Err(EinsumError::ContractedLabels { count: 0 })
        );
        // Two contracted labels need a nested sum.
        assert_eq!(
            einsum_plan("ijk,ijl->kl", &[2, 3, 4], &[2, 3, 5]),
            Err(EinsumError::ContractedLabels { count: 2 })
        );
        // A diagonal is not a strided read.
        assert_eq!(
            einsum_plan("ii,ij->ij", &[3, 3], &[3, 4]),
            Err(EinsumError::RepeatedLabel { label: 'i' })
        );
        // A label in one operand only and not in the output asks for a reduction
        // first, which would silently change the result if treated as contracted.
        assert_eq!(
            einsum_plan("ijx,jk->ik", &[2, 3, 7], &[3, 4]),
            Err(EinsumError::PartialReduction { label: 'x' })
        );
        // Extents must agree where a label is shared.
        assert_eq!(
            einsum_plan("ij,jk->ik", &[2, 3], &[5, 4]),
            Err(EinsumError::ExtentMismatch {
                label: 'j',
                a: 3,
                b: 5
            })
        );
        // Labels must match the tensors' ranks.
        assert!(matches!(
            einsum_plan("ij,jk->ik", &[2, 3, 4], &[3, 4]),
            Err(EinsumError::RankMismatch { operand: 0, .. })
        ));
        // An ellipsis stands for an unknown number of axes.
        assert_eq!(
            einsum_plan("...ij,...jk->...ik", &[2, 3], &[3, 4]),
            Err(EinsumError::Ellipsis)
        );
        // One operand, three operands, or no explicit output.
        assert_eq!(
            einsum_plan("ij->ji", &[2, 3], &[]),
            Err(EinsumError::NotTwoOperandForm)
        );
        assert_eq!(
            einsum_plan("ij,jk,kl->il", &[2, 3], &[3, 4]),
            Err(EinsumError::NotTwoOperandForm)
        );
        assert_eq!(
            einsum_plan("ij,jk", &[2, 3], &[3, 4]),
            Err(EinsumError::NotTwoOperandForm)
        );
        // Beyond the rank the kernel indexes.
        assert!(matches!(
            einsum_plan("abcde,ae->abcd", &[2, 2, 2, 2, 2], &[2, 2]),
            Err(EinsumError::RankTooHigh { .. })
        ));
    }

    /// Each output axis must be a distinct label. `ij,jk->iik` would otherwise plan a
    /// rank-3 output that reads A along its diagonal, an equation ONNX and NumPy both
    /// reject.
    #[test]
    fn test_rejects_a_repeated_output_label() {
        assert_eq!(
            einsum_plan("ij,jk->iik", &[2, 3], &[3, 4]),
            Err(EinsumError::RepeatedOutputLabel { label: 'i' })
        );
        // Not adjacent, and on a label only the right operand has.
        assert_eq!(
            einsum_plan("ij,jk->kik", &[2, 3], &[3, 4]),
            Err(EinsumError::RepeatedOutputLabel { label: 'k' })
        );
    }

    /// Labels are ASCII letters; anything else is reported as the character it is
    /// rather than as whatever structural fault it happens to cause later.
    #[test]
    fn test_rejects_a_label_that_is_not_a_letter() {
        // In either operand or the output.
        assert_eq!(
            einsum_plan("i1,1k->ik", &[2, 3], &[3, 4]),
            Err(EinsumError::InvalidLabel { label: '1' })
        );
        assert_eq!(
            einsum_plan("ij,jk->i_k", &[2, 3], &[3, 4]),
            Err(EinsumError::InvalidLabel { label: '_' })
        );
        // A single dot is not an ellipsis, and not a label either.
        assert_eq!(
            einsum_plan(".ij,jk->ik", &[1, 2, 3], &[3, 4]),
            Err(EinsumError::InvalidLabel { label: '.' })
        );
        // Uppercase letters are labels, distinct from their lowercase forms.
        let plan = einsum_plan("iJ,Jk->ik", &[2, 3], &[3, 4]).expect("uppercase label");
        assert_eq!(plan.out_shape, vec![2, 4]);
    }

    /// A second `->` is a malformed equation, not an output containing `-` and `>`.
    #[test]
    fn test_rejects_a_second_arrow() {
        assert_eq!(
            einsum_plan("ij,jk->ik->x", &[2, 3], &[3, 4]),
            Err(EinsumError::NotTwoOperandForm)
        );
    }

    /// The ellipsis is recognised after whitespace is dropped, so spacing it out does
    /// not turn it into three invalid labels.
    #[test]
    fn test_recognises_a_spaced_out_ellipsis() {
        assert_eq!(
            einsum_plan(". . .ij,. . .jk->. . .ik", &[2, 3], &[3, 4]),
            Err(EinsumError::Ellipsis)
        );
    }

    /// The contraction-count message ends at its last word.
    #[test]
    fn test_contracted_labels_message_has_no_trailing_space() {
        for count in [0, 2] {
            let msg = EinsumError::ContractedLabels { count }.to_string();
            assert_eq!(msg, msg.trim_end(), "{msg:?}");
        }
    }

    /// Whitespace is not significant, and an output label neither operand has is an
    /// error rather than an axis of extent 1.
    #[test]
    fn test_tolerates_spacing_and_catches_an_unknown_output_label() {
        assert_eq!(
            einsum_plan(" ij , jk -> ik ", &[2, 3], &[3, 4])
                .expect("spaces are not significant")
                .out_shape,
            vec![2, 4]
        );
        // `ij,jk->ikz` names z, which neither operand has. The equation has to be
        // otherwise sound to reach that check: in `ij,jk->iz` the real first fault is
        // that k is then in one operand only and not in the output.
        assert_eq!(
            einsum_plan("ij,jk->ikz", &[2, 3], &[3, 4]),
            Err(EinsumError::UnknownOutputLabel { label: 'z' })
        );
    }
}
