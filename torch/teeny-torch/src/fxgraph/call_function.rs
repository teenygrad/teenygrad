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

use egg::Id;
use regex::Regex;
use teeny_core::fxgraph::{FXGraph, lang::FxGraphLang};

use crate::{
    error::Error,
    fxgraph::{keyvalue::into_keyvalue, value::into_value},
    torch::CallFunction,
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

    let args: Vec<Id> = node
        .args()
        .unwrap_or_default()
        .iter()
        .map(|x| into_value(fxgraph, x))
        .collect::<Result<Vec<_>, Error>>()?
        .into_iter()
        .map(|x| fxgraph.add_value(x))
        .collect::<Vec<_>>();
    let keyvalues = node
        .kwargs()
        .unwrap_or_default()
        .iter()
        .map(|x| into_keyvalue(fxgraph, x))
        .collect::<Result<Vec<_>, Error>>()?;
    let kwargs = if keyvalues.is_empty() {
        None
    } else {
        Some(fxgraph.add_kwargs(keyvalues))
    };

    let name = node
        .name()
        .ok_or_else(|| Error::NoGraphNodeName(format!("{:?}", node)))?;

    let functions = get_functions();
    if let Some(handler) = functions.get(func_name) {
        handler(fxgraph, node, name, args, kwargs)?;
    } else {
        todo!("Unknown function: {func_name:?} {target:?}");
    }

    Ok(())
}

type NodeHandler = fn(&mut FXGraph, &CallFunction, &str, Vec<Id>, Option<Id>) -> Result<(), Error>;
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

fn mul(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    if args.len() != 2 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let args = [args[0], args[1]];

    fxgraph.add_operation(name, FxGraphLang::Mul(args));
    Ok(())
}

fn rsqrt(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    if args.len() != 1 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let args = [args[0]];

    fxgraph.add_operation(name, FxGraphLang::Rsqrt(args));
    Ok(())
}

fn linear(
    fxgraph: &mut FXGraph,
    _node: &CallFunction,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    fxgraph.add_operation(name, FxGraphLang::Linear(args));
    Ok(())
}

fn neg(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    if args.len() != 1 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    fxgraph.add_operation(name, FxGraphLang::Neg(args[0]));
    Ok(())
}

fn sym_sum(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    if args.len() != 1 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    fxgraph.add_operation(name, FxGraphLang::SymSum(args[0]));
    Ok(())
}

fn scaled_dot_product_attention(
    fxgraph: &mut FXGraph,
    _node: &CallFunction,
    name: &str,
    mut args: Vec<Id>,
    kwargs: Option<Id>,
) -> Result<(), Error> {
    if let Some(kwargs) = kwargs {
        args.push(kwargs);
    }

    fxgraph.add_operation(name, FxGraphLang::ScaledDotProductAttention(args));
    Ok(())
}

fn log_api_usage_once(
    _fxgraph: &mut FXGraph,
    _node: &CallFunction,
    _name: &str,
    _args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    // no-op
    Ok(())
}

fn getitem(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    if args.len() != 2 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let args = [args[0], args[1]];

    fxgraph.add_operation(name, FxGraphLang::GetItem(args));
    Ok(())
}

fn lazy_load_decompositions(
    fxgraph: &mut FXGraph,
    _node: &CallFunction,
    name: &str,
    _args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    fxgraph.add_operation(name, FxGraphLang::LazyLoadDecompositions([]));
    Ok(())
}

fn silu(
    fxgraph: &mut FXGraph,
    _node: &CallFunction,
    name: &str,
    mut args: Vec<Id>,
    kwargs: Option<Id>,
) -> Result<(), Error> {
    if let Some(kwargs) = kwargs {
        args.push(kwargs);
    }

    fxgraph.add_operation(name, FxGraphLang::Silu(args));
    Ok(())
}

fn aten_index(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    if args.len() != 2 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let args = [args[0], args[1]];
    fxgraph.add_operation(name, FxGraphLang::AtenIndex(args));
    Ok(())
}

fn add(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    if args.len() != 2 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let args = [args[0], args[1]];

    fxgraph.add_operation(name, FxGraphLang::Add(args));
    Ok(())
}

fn iadd(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    if args.len() != 2 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let args = [args[0], args[1]];

    fxgraph.add_operation(name, FxGraphLang::IAdd(args));
    Ok(())
}

fn arange(
    fxgraph: &mut FXGraph,
    _node: &CallFunction,
    name: &str,
    mut args: Vec<Id>,
    kwargs: Option<Id>,
) -> Result<(), Error> {
    if let Some(kwargs) = kwargs {
        args.push(kwargs);
    }

    fxgraph.add_operation(name, FxGraphLang::Arange(args));

    Ok(())
}

fn embedding(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    if args.len() != 7 {
        return Err(Error::GraphNodeMissingArgs(format!("args len: {node:?}")));
    }

    let args = [
        args[0], args[1], args[2], args[3], args[4], args[5], args[6],
    ];

    fxgraph.add_operation(name, FxGraphLang::Embedding(args));

    Ok(())
}

fn add_batch_dim(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    if args.len() != 3 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let args = [args[0], args[1], args[2]];

    fxgraph.add_operation(name, FxGraphLang::AddBatchDim(args));
    Ok(())
}

fn remove_batch_dim(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    if args.len() != 4 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let args = [args[0], args[1], args[2], args[3]];

    fxgraph.add_operation(name, FxGraphLang::RemoveBatchDim(args));
    Ok(())
}

fn vmap_decrement_nesting(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    if !args.is_empty() {
        return Err(Error::GraphNodeInvalidArgs(format!("{:?}", node)));
    }

    fxgraph.add_operation(name, FxGraphLang::VmapDecrementNesting([]));
    Ok(())
}

fn enter_autocast(
    fxgraph: &mut FXGraph,
    _node: &CallFunction,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    fxgraph.add_operation(name, FxGraphLang::EnterAutocast(args));
    Ok(())
}

fn exit_autocast(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    if args.len() != 1 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let args = [args[0]];

    fxgraph.add_operation(name, FxGraphLang::ExitAutocast(args));
    Ok(())
}

fn matmul(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    if args.len() != 2 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let args = [args[0], args[1]];

    fxgraph.add_operation(name, FxGraphLang::MatMul(args));
    Ok(())
}

fn vmap_increment_nesting(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    if args.len() != 2 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let args = [args[0], args[1]];

    fxgraph.add_operation(name, FxGraphLang::VmapIncrementNesting(args));
    Ok(())
}

fn cat(
    fxgraph: &mut FXGraph,
    node: &CallFunction,
    name: &str,
    mut args: Vec<Id>,
    kwargs: Option<Id>,
) -> Result<(), Error> {
    if args.len() != 1 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    if let Some(kwargs) = kwargs {
        args.push(kwargs);
    }

    fxgraph.add_operation(name, FxGraphLang::Cat(args));
    Ok(())
}
