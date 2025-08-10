/*
 * Copyright (c) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

use teeny_cache::DynamicCache;
use teeny_core::dtype::DtypeEnum;
use teeny_core::graph::ops::Op;
use teeny_core::graph::{NodeOp, NodeRef, arange, scalar};
use teeny_core::num::bool::Bool;
use teeny_core::slice;

use crate::error::{Error, Result};
use crate::transformer::model::qwen::qwen3::attention::Attention;
use crate::transformer::model::qwen::qwen3::qwen3_config::Qwen3Config;

pub type MaskFunction<'data> =
    Box<dyn Fn(&Qwen3Config, usize, usize, &NodeRef<'data>, usize) -> NodeRef<'data> + 'data>;

#[allow(clippy::too_many_arguments)]
pub fn create_causal_mask<'data>(
    config: &Qwen3Config,
    input_embeds: &NodeRef<'data>,
    attention_mask: &Option<NodeRef<'data>>,
    cache_position: &NodeRef<'data>,
    past_key_values: &Option<DynamicCache>,
    position_ids: Option<NodeRef<'data>>,
    or_mask_function: Option<MaskFunction<'data>>,
    and_mask_function: Option<MaskFunction<'data>>,
) -> Result<Option<NodeRef<'data>>> {
    let layer_idx = 0; // AXM TODO: Use cache to determine the layer idx

    let (early_exit, attention_mask, packed_sequence_mask, kv_length, kv_offset) =
        preprocess_mask_arguments(
            config,
            input_embeds,
            attention_mask,
            cache_position,
            past_key_values,
            &position_ids,
            layer_idx,
        )?;

    if early_exit {
        return Ok(attention_mask.clone());
    }

    let batch_size = input_embeds.shape()?.dims()[0];
    let dtype = input_embeds.dtype();
    let mut mask_factory_function: MaskFunction = Box::new(causal_mask_function);
    let mask_interface = match config.attn_implementation {
        Attention::FlexAttention => flash_attention_mask,
        Attention::FlashAttention2 => todo!(), // flex_attention_mask,
    };

    let mut allow_is_causal_skip = past_key_values
        .as_ref()
        .map(|cache| !cache.is_compileable())
        .unwrap_or(true);

    if let Some(packed_sequence_mask) = packed_sequence_mask {
        mask_factory_function = and_masks(vec![
            mask_factory_function,
            packed_sequence_mask_function(packed_sequence_mask),
        ]);
        allow_is_causal_skip = false;
    }

    if let Some(mask_function) = or_mask_function {
        mask_factory_function = or_masks(vec![mask_factory_function, mask_function]);
        allow_is_causal_skip = false;
    }

    if let Some(mask_function) = and_mask_function {
        mask_factory_function = and_masks(vec![mask_factory_function, mask_function]);
        allow_is_causal_skip = false;
    }

    let causal_mask = mask_interface(
        batch_size,
        cache_position,
        kv_length.unwrap_or(0),
        kv_offset.unwrap_or(0),
        Some(mask_factory_function),
        attention_mask.clone(),
        allow_is_causal_skip,
        dtype,
    );

    Ok(causal_mask)
}

#[allow(clippy::too_many_arguments)]
pub fn create_sliding_window_causal_mask<'data>(
    _config: &Qwen3Config,
    _input_embeds: &NodeRef<'data>,
    _attention_mask: &Option<NodeRef<'data>>,
    _cache_position: &NodeRef<'data>,
    _past_key_values: &Option<DynamicCache>,
    _position_ids: Option<NodeRef<'data>>,
    _or_mask_function: Option<MaskFunction>,
    _and_mask_function: Option<MaskFunction>,
) -> Result<Option<NodeRef<'data>>> {
    todo!()
}

#[allow(clippy::too_many_arguments)]
pub fn _flex_attention_mask<'data>(
    _batch_size: usize,
    cache_position: &NodeRef<'data>,
    kv_length: usize,
    kv_offset: usize,
    mask_function: MaskFunction<'data>,
    attention_mask: Option<NodeRef<'data>>,
    _allow_is_causal_skip: bool,
    _dtype: DtypeEnum,
) -> Result<Option<NodeRef<'data>>> {
    let _q_length = cache_position.shape()?.dims()[0];
    let _q_offset = cache_position.index(vec![scalar(0)]);

    let _mask_function = if let Some(attention_mask) = attention_mask {
        let padding_mask = prepare_padding_mask(Some(attention_mask), kv_length, kv_offset, false)?
            .ok_or(Error::ModelError("Padding mask is None".to_string()))?;
        and_masks(vec![mask_function, padding_mask_function(padding_mask)])
    } else {
        mask_function
    };

    // let mask_function = add_offsets_to_mask_function(mask_function, q_offset, kv_offset);

    // let block_mask = create_block_mask(
    //     mask_function,
    //     Some(batch_size),
    //     None,
    //     q_length,
    //     kv_length,
    //     &[batch_size],
    // );

    // Ok(block_mask)
    todo!()
}

enum ModificationType<'data> {
    ScoreMod(
        Box<
            dyn Fn(
                    NodeRef<'data>,
                    NodeRef<'data>,
                    NodeRef<'data>,
                    NodeRef<'data>,
                    NodeRef<'data>,
                ) -> NodeRef<'data>
                + 'data,
        >,
    ),
    MaskMod(
        Box<
            dyn Fn(
                    NodeRef<'data>,
                    NodeRef<'data>,
                    NodeRef<'data>,
                    NodeRef<'data>,
                    NodeRef<'data>,
                ) -> NodeRef<'data>
                + 'data,
        >,
    ),
}

#[allow(non_snake_case)]
fn _create_mask<'data>(
    _mod_fn: ModificationType<'data>,
    B: Option<usize>,
    H: Option<usize>,
    Q_LEN: usize,
    KV_LEN: usize,
) -> NodeRef<'data> {
    let B = B.unwrap_or(1);
    let H = H.unwrap_or(1);
    let _b = arange(0, B, 1);
    let _h = arange(0, H, 1);
    let _m = arange(0, Q_LEN, 1);
    let _n = arange(0, KV_LEN, 1);

    todo!()
    // let mask = match &mod_fn {
    //     ModificationType::ScoreMod(score_mod) => {
    //         let prefix = &[Some(0)];
    //         let suffix = &[];
    //         let score_mod = vmap_for_bhqkv(mod_fn, &[Some(0)], &[], 0, false);
    //         let out = score_mod(
    //             zeros(&[B, H, Q_LEN, KV_LEN], DtypeEnum::Default),
    //             b,
    //             h,
    //             m,
    //             n,
    //         );
    //         graph::r#where(
    //             graph::isneginf(out),
    //             scalar(Bool(false)),
    //             scalar(Bool(true)),
    //         )
    //     }

    //     ModificationType::MaskMod(mask_mod) => {
    //         let mask_mod = vmap_for_bhqkv(mod_fn, &[], &[], 0, false);
    //         mask_mod(b, h, m, n)
    //     }
    // };

    // Ok(mask)
}

fn _vmap_for_bhqkv<'data>(
    _mod_fn: ModificationType<'data>,
    _prefix: &[Option<usize>],
    _suffix: &[Option<usize>],
    _out_dims: usize,
    group_dim: bool,
) -> MaskFunction<'data> {
    let mut dimensions = vec![];
    dimensions.push(&[None, None, None, Some(0)]);
    dimensions.push(&[None, None, Some(0), None]);
    dimensions.push(&[None, Some(0), None, None]);

    if group_dim {
        dimensions.push(&[None, Some(0), None, None]);
    }

    dimensions.push(&[Some(0), None, None, None]);

    // let mut mask_fn = mod_fn;

    // for dims in dimensions {
    //     let in_dims = prefix
    //         .clone()
    //         .iter()
    //         .copied()
    //         .chain(dims.iter().copied())
    //         .chain(suffix.iter().copied())
    //         .collect::<Vec<_>>();

    //     mask_fn = graph::vmap(mask_fn, &in_dims, out_dims);
    // }

    // Ok(mask_fn)
    todo!()
}

#[allow(non_snake_case)]
fn _create_block_mask<'data>(
    _mask_mod: ModificationType<'data>,
    B: Option<usize>,
    H: Option<usize>,
    _Q_LEN: usize,
    _KV_LEN: usize,
    BLOCK_SIZE: &[usize],
) {
    let _B = B.unwrap_or(1);
    let _H = H.unwrap_or(1);

    let (_Q_BLOCK_SIZE, _KV_BLOCK_SIZE) = if BLOCK_SIZE.len() == 1 {
        (BLOCK_SIZE[0], BLOCK_SIZE[0])
    } else {
        (BLOCK_SIZE[0], BLOCK_SIZE[1])
    };

    // let mask_tensor = create_mask(mask_mod, B, H, Q_LEN, KV_LEN);
    // let (partial_block_mask, full_block_mask) =
    //     convert_mask_to_block_mask(mask_tensor, Q_BLOCK_SIZE, KV_BLOCK_SIZE, true);

    // let block_mask = create_sparse_block_from_block_mask(
    //     (partial_block_mask, full_block_mask),
    //     mask_mod,
    //     (Q_LEN, KV_LEN),
    //     Q_BLOCK_SIZE,
    //     KV_BLOCK_SIZE,
    // );

    // Ok(block_mask)

    todo!()
}

fn _add_offsets_to_mask_function<'data>(
    mask_function: MaskFunction<'data>,
    q_offset: NodeRef<'data>,
    kv_offset: usize,
) -> MaskFunction<'data> {
    let inner_mask = move |_config: &Qwen3Config,
                           batch_idx: usize,
                           _head_idx: usize,
                           q_idx: &NodeRef<'data>,
                           kv_idx: usize|
          -> NodeRef<'data> {
        let q = q_idx.clone() + q_offset.clone();
        mask_function(_config, batch_idx, _head_idx, &q, kv_idx + kv_offset)
    };

    Box::new(inner_mask)
}

fn padding_mask_function<'data>(padding_mask: NodeRef<'data>) -> MaskFunction<'data> {
    let inner_mask = move |_config: &Qwen3Config,
                           batch_idx: usize,
                           _head_idx: usize,
                           _q_idx: &NodeRef<'data>,
                           kv_idx: usize| {
        padding_mask.index(vec![scalar(batch_idx), scalar(kv_idx)])
    };

    Box::new(inner_mask)
}

fn prepare_padding_mask<'data>(
    attention_mask: Option<NodeRef<'data>>,
    kv_length: usize,
    kv_offset: usize,
    slice: bool,
) -> Result<Option<NodeRef<'data>>> {
    let mut local_padding_mask = attention_mask.clone();
    if let Some(attention_mask) = attention_mask {
        let padding_length = kv_length + kv_offset - attention_mask.shape()?.last();
        if padding_length > 0 {
            local_padding_mask = Some(attention_mask.pad(&[0, padding_length]));
        }
        if slice {
            let mask_indices = arange(kv_offset, kv_offset + kv_length, 1);
            local_padding_mask =
                local_padding_mask.map(|mask| mask.slice(&slice!(.., mask_indices)));
        }
    }

    Ok(local_padding_mask)
}
//     attention_mask: Optional[torch.Tensor], kv_length: int, kv_offset: int, _slice: bool = True
// ) -> Optional[torch.Tensor]:
//     """
//     From the 2D attention mask, prepare the correct padding mask to use by potentially padding it, and slicing
//     according to the `kv_offset` if `_slice` is `True`.
//     """
//     local_padding_mask = attention_mask
//     if attention_mask is not None:
//         # Pad it if necesary
//         if (padding_length := kv_length + kv_offset - attention_mask.shape[-1]) > 0:
//             local_padding_mask = torch.nn.functional.pad(attention_mask, (0, padding_length))
//         # For flex, we should not slice them, only use an offset
//         if _slice:
//             # Equivalent to: `local_padding_mask = attention_mask[:, kv_offset : kv_offset + kv_length]`,
//             # but without data-dependent slicing (i.e. torch.compile friendly)
//             mask_indices = torch.arange(kv_length, device=local_padding_mask.device)
//             mask_indices += kv_offset
//             local_padding_mask = local_padding_mask[:, mask_indices]
//     return local_padding_mask

#[allow(clippy::type_complexity)]
pub fn preprocess_mask_arguments<'a, 'data>(
    _config: &Qwen3Config,
    input_embeds: &NodeRef<'data>,
    attention_mask: &'a Option<NodeRef<'data>>,
    _cache_position: &NodeRef<'data>,
    past_key_values: &Option<DynamicCache>,
    position_ids: &Option<NodeRef<'data>>,
    _layer_idx: usize,
) -> Result<(
    bool,
    &'a Option<NodeRef<'data>>,
    Option<NodeRef<'data>>,
    Option<usize>,
    Option<usize>,
)> {
    if let Some(mask) = attention_mask {
        match &mask.0.op {
            NodeOp::TensorF32(t) => {
                if t.shape()?.dims().len() == 4 {
                    return Ok((true, attention_mask, None, None, None));
                }
            }
            NodeOp::TensorBF16(t) => {
                if t.shape()?.dims().len() == 4 {
                    return Ok((true, attention_mask, None, None, None));
                }
            }
            _ => {
                return Err(
                    Error::ModelError("Attention mask must be a 4D tensor.".to_string()).into(),
                );
            }
        }
    }

    let mut packed_sequence_mask = None;
    if position_ids.is_some() && attention_mask.is_none() && past_key_values.is_none() {
        let batch_size = input_embeds.shape()?.dims()[0];
        let mut position_ids = position_ids.as_ref().unwrap().clone();
        if batch_size != position_ids.shape()?.dims()[0] {
            position_ids = position_ids.expand(&[batch_size as isize, -1]);
        }
        packed_sequence_mask = Some(find_packed_sequence_indices(position_ids)?);
    }

    // AXM TODO: Implement this, but it needs the cache
    let kv_length = None;
    let kv_offset = None;

    Ok((
        false,
        attention_mask,
        packed_sequence_mask,
        kv_length,
        kv_offset,
    ))
}

fn find_packed_sequence_indices<'data>(position_ids: NodeRef<'data>) -> Result<NodeRef<'data>> {
    let first_dummy_value = position_ids.slice(&slice!(.., ..1)) - scalar(1);
    let position_diff = position_ids.diff(&first_dummy_value, -1);
    Ok((position_diff.neq(&scalar(1))).cumsum(-1))
}

#[allow(clippy::too_many_arguments)]
pub fn flash_attention_mask<'data>(
    _batch_size: usize,
    _cache_position: &NodeRef<'data>,
    _kv_length: usize,
    _kv_offset: usize,
    _mask_function: Option<MaskFunction>,
    _attention_mask: Option<NodeRef<'data>>,
    _allow_is_causal_skip: bool,
    _dtype: DtypeEnum,
) -> Option<NodeRef<'data>> {
    let _mask_function = _mask_function.unwrap_or(Box::new(causal_mask_function));
    todo!()
}

pub fn causal_mask_function<'data>(
    _config: &Qwen3Config,
    _batch_idx: usize,
    _head_idx: usize,
    q_idx: &NodeRef<'data>,
    kv_idx: usize,
) -> NodeRef<'data> {
    scalar(kv_idx).leq(q_idx)
}

pub fn or_masks<'data>(mask_functions: Vec<MaskFunction<'data>>) -> MaskFunction<'data> {
    let and_mask = move |_config: &Qwen3Config,
                         _batch_idx: usize,
                         _head_idx: usize,
                         q_idx: &NodeRef<'data>,
                         kv_idx: usize|
          -> NodeRef<'data> {
        mask_functions
            .iter()
            .fold(scalar(Bool(false)), |acc, mask_function| {
                acc | mask_function(_config, _batch_idx, _head_idx, q_idx, kv_idx)
            })
    };
    Box::new(and_mask)
}

pub fn and_masks<'data, T: IntoIterator<Item = MaskFunction<'data>>>(
    mask_functions: T,
) -> MaskFunction<'data> {
    let mask_functions = mask_functions.into_iter().collect::<Vec<_>>();
    let and_mask = move |_config: &Qwen3Config,
                         _batch_idx: usize,
                         _head_idx: usize,
                         q_idx: &NodeRef<'data>,
                         kv_idx: usize|
          -> NodeRef<'data> {
        mask_functions
            .iter()
            .fold(scalar(Bool(true)), |acc, mask_function| {
                acc & mask_function(_config, _batch_idx, _head_idx, q_idx, kv_idx)
            })
    };
    Box::new(and_mask)
}

pub fn packed_sequence_mask_function<'data>(
    packed_sequence_mask: NodeRef<'data>,
) -> MaskFunction<'data> {
    let inner_mask = move |_config: &Qwen3Config,
                           batch_idx: usize,
                           _head_idx: usize,
                           q_idx: &NodeRef<'data>,
                           kv_idx: usize|
          -> NodeRef<'data> {
        packed_sequence_mask
            .index(vec![scalar(batch_idx), q_idx.clone()])
            .eq(&packed_sequence_mask.index(vec![scalar(batch_idx), scalar(kv_idx)]))
    };

    Box::new(inner_mask)
}
