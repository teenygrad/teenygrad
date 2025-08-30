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

use std::{collections::HashMap, sync::OnceLock};

use regex::Regex;
use teeny_core::fxgraph::{FXGraph, lang::FxGraphLang};

use crate::{
    error::Error,
    fxgraph::value::{into_value, node_value},
    torch::{CallFunction, ValueWrapper},
};

pub fn call_function<'a>(fxgraph: &'a mut FXGraph, node: &CallFunction<'a>) -> Result<(), Error> {
    let target = node
        .target()
        .ok_or_else(|| Error::NoGraphNodeTarget(format!("{node:?}")))?;
    let pat = r#"^(?:<function\s+|<built-in\s+(?:method|function)\s+)([a-zA-Z_][a-zA-Z0-9_]*)"#;
    let re = Regex::new(pat).unwrap();
    let func_name = re
        .captures(target)
        .and_then(|x| x.get(1))
        .map(|x| x.as_str())
        .unwrap_or(target);

    let args = node
        .args()
        .ok_or_else(|| Error::NoGraphNodeArgs(format!("args: {node:?}")))?
        .iter()
        .collect::<Vec<_>>();

    let name = node
        .name()
        .ok_or_else(|| Error::NoGraphNodeName(format!("{:?}", node)))?;

    let functions = get_functions();
    if let Some(handler) = functions.get(func_name) {
        handler(fxgraph, node, name, &args)?;
    } else {
        todo!("Unknown function: {func_name:?} {target:?}");
    }

    Ok(())
}

type NodeHandler = fn(&mut FXGraph, &CallFunction, &str, &[ValueWrapper]) -> Result<(), Error>;
static FUNCS: OnceLock<HashMap<String, NodeHandler>> = OnceLock::new();

// Initialize the functions HashMap
fn get_functions() -> &'static HashMap<String, NodeHandler> {
    FUNCS.get_or_init(|| {
        HashMap::from([
            ("embedding".to_string(), embedding as NodeHandler),
            ("_enter_autocast".to_string(), enter_autocast as NodeHandler),
            ("_exit_autocast".to_string(), exit_autocast as NodeHandler),
            (
                "lazy_load_decompositions".to_string(),
                lazy_load_decompositions as NodeHandler,
            ),
            ("silu".to_string(), silu as NodeHandler),
            ("getitem".to_string(), getitem as NodeHandler),
            ("sym_sum".to_string(), sym_sum as NodeHandler),
            ("aten.index".to_string(), aten_index as NodeHandler),
            ("arange".to_string(), arange as NodeHandler),
            ("iadd".to_string(), iadd as NodeHandler),
            ("add".to_string(), add as NodeHandler),
            ("matmul".to_string(), matmul as NodeHandler),
            (
                "_vmap_increment_nesting".to_string(),
                vmap_increment_nesting as NodeHandler,
            ),
            (
                "_vmap_decrement_nesting".to_string(),
                vmap_decrement_nesting as NodeHandler,
            ),
            ("_add_batch_dim".to_string(), add_batch_dim as NodeHandler),
            (
                "_remove_batch_dim".to_string(),
                remove_batch_dim as NodeHandler,
            ),
            ("cat".to_string(), cat as NodeHandler),
            ("mul".to_string(), mul as NodeHandler),
            ("rsqrt".to_string(), rsqrt as NodeHandler),
            ("linear".to_string(), linear as NodeHandler),
            ("neg".to_string(), neg as NodeHandler),
            ("sym_sum".to_string(), sym_sum as NodeHandler),
            (
                "scaled_dot_product_attention".to_string(),
                scaled_dot_product_attention as NodeHandler,
            ),
            (
                "_log_api_usage_once".to_string(),
                log_api_usage_once as NodeHandler,
            ),
        ])
    })
}

fn mul<'a>(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: &[ValueWrapper<'a>],
) -> Result<(), Error> {
    todo!("mul: {:?}", node);
    // if args.len() != 2 {
    //     return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    // }

    // let arg1 = find_or_create(fxgraph, args[0]);
    // let arg2 = find_or_create(fxgraph, args[1]);

    // fxgraph.add_operation(name, FxGraphLang::Mul([arg1, arg2]));

    // Ok(())
}

fn rsqrt<'a>(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: &[ValueWrapper<'a>],
) -> Result<(), Error> {
    todo!("rsqrt: {:?}", node);
    // if args.len() != 1 {
    //     return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    // }

    // let arg1 = find_or_create(fxgraph, args[0]);
    // fxgraph.add_operation(name, FxGraphLang::Rsqrt([arg1]));

    // Ok(())
}

fn linear<'a>(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: &[ValueWrapper<'a>],
) -> Result<(), Error> {
    todo!("linear: {:?}", node);
    // if args.len() != 2 {
    //     return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    // }

    // let arg1 = find_or_create(fxgraph, args[0]);
    // let arg2 = find_or_create(fxgraph, args[1]);

    // fxgraph.add_operation(name, FxGraphLang::Linear([arg1, arg2]));

    // Ok(())
}

fn neg<'a>(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: &[ValueWrapper<'a>],
) -> Result<(), Error> {
    todo!("neg: {:?}", node);
    // if args.len() != 1 {
    //     return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    // }

    // let arg1 = find_or_create(fxgraph, args[0]);
    // fxgraph.add_operation(name, FxGraphLang::Neg(arg1));

    // Ok(())
}

fn sym_sum<'a>(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: &[ValueWrapper<'a>],
) -> Result<(), Error> {
    todo!("sym_sum: {:?}", node);
    // if args.len() != 1 {
    //     return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    // }

    // if args[0].starts_with("[") && args[0].ends_with("]") {
    //     let args = args[0][1..args[0].len() - 1]
    //         .split(",")
    //         .map(|x| find_or_create(fxgraph, x))
    //         .collect::<Vec<_>>();
    //     fxgraph.add_operation(name, FxGraphLang::SymSum(args));
    // } else {
    //     return Err(Error::GraphNodeInvalidArgs(format!("sym_sum - {:?}", node)));
    // }

    // Ok(())
}

fn scaled_dot_product_attention<'a>(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: &[ValueWrapper<'a>],
) -> Result<(), Error> {
    todo!("scaled_dot_product_attention: {:?}", node);
    // if args.len() != 3 {
    //     return Err(Error::GraphNodeMissingArgs(format!(
    //         "arg len  {} - {:?}",
    //         args.len(),
    //         node
    //     )));
    // }

    // let arg1 = find_or_create(fxgraph, args[0]);
    // let arg2 = find_or_create(fxgraph, args[1]);
    // let arg3 = find_or_create(fxgraph, args[2]);

    // let attn_mask = find_or_create(
    //     fxgraph,
    //     &find_kw_arg::<String>(node, "attn_mask")?.ok_or(Error::GraphNodeMissingArgs(format!(
    //         "attn_mask - {:?}",
    //         node
    //     )))?,
    // );

    // let dropout_p =
    //     fxgraph.add_operation(
    //         &fxgraph.unique_name(),
    //         const_f32(find_kw_arg::<f32>(node, "dropout_p")?.ok_or(
    //             Error::GraphNodeMissingArgs(format!("dropout_p - {:?}", node)),
    //         )?),
    //     );

    // let scale = find_or_create(
    //     fxgraph,
    //     &find_kw_arg::<String>(node, "scale")?
    //         .ok_or(Error::GraphNodeMissingArgs(format!("scale - {:?}", node)))?,
    // );

    // let is_causal = fxgraph.add_operation(
    //     &fxgraph.unique_name(),
    //     const_bool(&find_kw_arg::<String>(node, "is_causal")?.ok_or(
    //         Error::GraphNodeMissingArgs(format!("is_causal - {:?}", node)),
    //     )?),
    // );

    // fxgraph.add_operation(
    //     name,
    //     FxGraphLang::ScaledDotProductAttention([
    //         arg1, arg2, arg3, attn_mask, dropout_p, scale, is_causal,
    //     ]),
    // );

    // Ok(())
}

fn log_api_usage_once<'a>(
    _fxgraph: &mut FXGraph,
    _node: &CallFunction,
    _name: &str,
    _args: &[ValueWrapper<'a>],
) -> Result<(), Error> {
    // no-op
    Ok(())
}

fn getitem<'a>(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: &[ValueWrapper<'a>],
) -> Result<(), Error> {
    todo!("getitem: {:?}", node);
}

fn lazy_load_decompositions<'a>(
    fxgraph: &mut FXGraph,
    _node: &CallFunction,
    name: &str,
    _args: &[ValueWrapper<'a>],
) -> Result<(), Error> {
    todo!("lazy_load_decompositions: {:?}", _node);
    // fxgraph.add_operation(name, FxGraphLang::LazyLoadDecompositions([]));
    // Ok(())
}

fn silu<'a>(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: &[ValueWrapper<'a>],
) -> Result<(), Error> {
    todo!("silu: {:?}", node);
    // if args.len() != 1 {
    //     return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    // }

    // let arg1 = find_or_create(fxgraph, args[0]);
    // let inplace =
    //     const_bool(&find_kw_arg::<String>(node, "inplace")?.unwrap_or("false".to_string()));
    // let arg2 = fxgraph.add_operation(&fxgraph.unique_name(), inplace);

    // fxgraph.add_operation(name, FxGraphLang::Silu([arg1, arg2]));

    // Ok(())
}

fn aten_index<'a>(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: &[ValueWrapper<'a>],
) -> Result<(), Error> {
    todo!("aten_index: {:?}", node);
    // if args.len() != 2 {
    //     return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    // }

    // let arg1 = find_or_create(fxgraph, args[0]);
    // let arg2 = find_or_create(fxgraph, args[1]);

    // fxgraph.add_operation(name, FxGraphLang::AtenIndex([arg1, arg2]));

    // Ok(())
}

fn add<'a>(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: &[ValueWrapper<'a>],
) -> Result<(), Error> {
    todo!("add: {:?}", node);
    // if args.len() != 2 {
    //     return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    // }

    // let arg1 = find_or_create(fxgraph, args[0]);
    // let arg2 = find_or_create(fxgraph, args[1]);

    // fxgraph.add_operation(name, FxGraphLang::Add([arg1, arg2]));

    // Ok(())
}

fn iadd<'a>(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: &[ValueWrapper<'a>],
) -> Result<(), Error> {
    todo!("iadd: {:?}", node);
    // if args.len() != 2 {
    //     return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    // }

    // let arg1 = find_or_create(fxgraph, args[0]);
    // let arg2 = find_or_create(fxgraph, args[1]);
    // fxgraph.add_operation(name, FxGraphLang::IAdd([arg1, arg2]));
    // Ok(())
}

fn arange<'a>(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: &[ValueWrapper<'a>],
) -> Result<(), Error> {
    todo!("arange: {:?}", node);
    // let start;
    // let end;
    // let step;

    // match args.len() {
    //     1 => {
    //         start = fxgraph.add_operation(&fxgraph.unique_name(), const_i64(0));
    //         end = find_or_create(fxgraph, args[0]);
    //         step = fxgraph.add_operation(&fxgraph.unique_name(), const_i64(1));
    //     }
    //     2 => {
    //         start = find_or_create(fxgraph, args[0]);
    //         end = find_or_create(fxgraph, args[1]);
    //         step = fxgraph.add_operation(&fxgraph.unique_name(), const_i64(1));
    //     }
    //     3 => {
    //         start = find_or_create(fxgraph, args[0]);
    //         end = find_or_create(fxgraph, args[1]);
    //         step = find_or_create(fxgraph, args[2]);
    //     }
    //     _ => {
    //         return Err(Error::GraphNodeMissingArgs(format!("args len: {node:?}")));
    //     }
    // }

    // fxgraph.add_operation(name, FxGraphLang::Arange([start, end, step]));

    // Ok(())
}

fn embedding<'a>(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: &[ValueWrapper<'a>],
) -> Result<(), Error> {
    if args.len() != 7 {
        return Err(Error::GraphNodeMissingArgs(format!("args len: {node:?}")));
    }

    let input_ids = node_value(fxgraph, args[0])?;
    let weight = node_value(fxgraph, args[1])?;

    let padding_idx = FxGraphLang::Value(into_value(fxgraph, args[2])?);
    let max_norm = FxGraphLang::Value(into_value(fxgraph, args[3])?);
    let norm_type = FxGraphLang::Value(into_value(fxgraph, args[4])?);
    let scale_grad_by_freq = FxGraphLang::Value(into_value(fxgraph, args[5])?);
    let sparse = FxGraphLang::Value(into_value(fxgraph, args[6])?);

    let padding_idx = fxgraph.add_operation(&fxgraph.unique_name(), padding_idx);
    let max_norm = fxgraph.add_operation(&fxgraph.unique_name(), max_norm);
    let norm_type = fxgraph.add_operation(&fxgraph.unique_name(), norm_type);
    let scale_grad_by_freq = fxgraph.add_operation(&fxgraph.unique_name(), scale_grad_by_freq);
    let sparse = fxgraph.add_operation(&fxgraph.unique_name(), sparse);

    fxgraph.add_operation(
        name,
        FxGraphLang::Embedding([
            input_ids,
            weight,
            padding_idx,
            max_norm,
            norm_type,
            scale_grad_by_freq,
            sparse,
        ]),
    );

    Ok(())
}

fn add_batch_dim<'a>(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: &[ValueWrapper<'a>],
) -> Result<(), Error> {
    if args.len() != 3 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    todo!("add_batch_dim: {:?}", node);
    // let arg1 = find_or_create(fxgraph, args[0]);
    // let arg2 = find_or_create(fxgraph, args[1]);
    // let arg3 = find_or_create(fxgraph, args[2]);
    // fxgraph.add_operation(name, FxGraphLang::AddBatchDim([arg1, arg2, arg3]));

    // Ok(())
}

fn remove_batch_dim<'a>(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: &[ValueWrapper<'a>],
) -> Result<(), Error> {
    if args.len() != 4 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    todo!("remove_batch_dim: {:?}", node);
    // let arg1 = find_or_create(fxgraph, args[0]);
    // let arg2 = find_or_create(fxgraph, args[1]);
    // let arg3 = find_or_create(fxgraph, args[2]);
    // let arg4 = find_or_create(fxgraph, args[3]);
    // fxgraph.add_operation(name, FxGraphLang::RemoveBatchDim([arg1, arg2, arg3, arg4]));

    // Ok(())
}

fn vmap_decrement_nesting<'a>(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: &[ValueWrapper<'a>],
) -> Result<(), Error> {
    if !args.is_empty() {
        return Err(Error::GraphNodeInvalidArgs(format!("{:?}", node)));
    }

    todo!("vmap_decrement_nesting: {:?}", node);
    // fxgraph.add_operation(name, FxGraphLang::VmapDecrementNesting([]));

    // Ok(())
}

fn enter_autocast<'a>(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: &[ValueWrapper<'a>],
) -> Result<(), Error> {
    if args.len() != 2 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    todo!("enter_autocast: {:?}", node);
    // let arg1 = fxgraph.add_operation(name, const_string(args[0]));
    // let arg2 = find_or_create(fxgraph, args[1]);
    // fxgraph.add_operation(name, FxGraphLang::EnterAutocast([arg1, arg2]));

    // Ok(())
}

fn exit_autocast<'a>(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: &[ValueWrapper<'a>],
) -> Result<(), Error> {
    if args.len() != 1 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    todo!("exit_autocast: {:?}", node);
    // let arg1 = fxgraph
    //     .get_node(args[0])
    //     .ok_or_else(|| Error::GraphNodeNotFound(format!("{:?}", node)))?;

    // fxgraph.add_operation(name, FxGraphLang::ExitAutocast([arg1]));

    // Ok(())
}

fn matmul<'a>(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: &[ValueWrapper<'a>],
) -> Result<(), Error> {
    if args.len() != 2 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    todo!("matmul: {:?}", node);
    // let arg1 = find_or_create(fxgraph, args[0]);
    // let arg2 = find_or_create(fxgraph, args[1]);

    // fxgraph.add_operation(name, FxGraphLang::MatMul([arg1, arg2]));

    // Ok(())
}

fn vmap_increment_nesting<'a>(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: &[ValueWrapper<'a>],
) -> Result<(), Error> {
    if args.len() != 2 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    todo!("vmap_increment_nesting: {:?}", node);
    // let arg1 = find_or_create(fxgraph, args[0]);
    // let arg2 = find_or_create(fxgraph, args[1]);
    // fxgraph.add_operation(name, FxGraphLang::VmapIncrementNesting([arg1, arg2]));

    // Ok(())
}

fn cat<'a>(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: &[ValueWrapper<'a>],
) -> Result<(), Error> {
    if args.len() != 1 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    todo!("cat: {:?}", node);
    // let arg1 = find_or_create(fxgraph, args[0]);

    // let dim = find_kw_arg::<i64>(node, "dim")?.unwrap_or(0);
    // let dim = fxgraph.add_operation(&fxgraph.unique_name(), const_i64(dim));
    // // let arg2 = fxgraph.add_operation(&fxgraph.unique_name(), const_kv("dim", dim));

    // fxgraph.add_operation(name, FxGraphLang::Cat([arg1, arg2]));

    // Ok(())
}
