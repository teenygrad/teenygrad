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

use crate::{error::Error, graph::Graph};
use std::convert::TryFrom;
use std::sync::OnceLock;
use std::{collections::HashMap, str::FromStr};

use egg::Id;

use regex::Regex;
use teeny_compiler::fxgraph;
use teeny_core::fxgraph::lang::const_f32;
use teeny_core::fxgraph::{
    FXGraph,
    lang::{FxGraphLang, const_bool, const_i64, const_kv, const_string},
};

use crate::graph::{Node, OpType};

impl<'a> TryFrom<Graph<'a>> for FXGraph {
    type Error = Error;

    fn try_from(graph: Graph<'a>) -> Result<Self, Self::Error> {
        let mut fxgraph = FXGraph::new();

        // Safely access nodes with error handling
        let nodes = graph.nodes().ok_or(Error::NoGraphNodes)?;

        for node in nodes {
            let _name = node
                .name()
                .ok_or_else(|| Error::NoGraphNodeName(format!("{node:?}")))?;
            let op = node.op();

            match op {
                OpType::placeholder => {
                    handle_placeholder(&mut fxgraph, &node)?;
                }
                OpType::call_function => {
                    call_function(&mut fxgraph, &node)?;
                }
                OpType::call_method => {
                    call_method(&mut fxgraph, &node)?;
                }
                OpType::output => {
                    handle_output(&mut fxgraph, &node)?;
                }
                _ => {
                    println!("Unknown op: {node:?}");
                    // return Err(Error::UnsupportedOp(op));
                }
            }
        }

        Ok(fxgraph)
    }
}

// Static HashMap mapping function names to their handler functions
type NodeHandler = fn(&mut FXGraph, &Node, &str, &[&str]) -> Result<(), Error>;
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

fn find_or_create(fxgraph: &mut FXGraph, name: &str) -> Id {
    let node = fxgraph.get_node(name);
    if let Some(id) = node {
        return id;
    }

    let v = name.parse::<i64>();
    if let Ok(v) = v {
        return fxgraph.add_operation(&fxgraph.unique_name(), const_i64(v));
    }

    let v = name.to_lowercase().parse::<bool>();
    if let Ok(v) = v {
        return fxgraph.add_operation(&fxgraph.unique_name(), const_bool(&v.to_string()));
    }

    if name.starts_with("(") && name.ends_with(")") {
        let args = name[1..name.len() - 1].split(",").collect::<Vec<_>>();
        let args = args
            .iter()
            .map(|x| find_or_create(fxgraph, x))
            .collect::<Vec<_>>();
        return fxgraph.add_operation(&fxgraph.unique_name(), FxGraphLang::Tuple(args));
    }

    fxgraph.add_operation(name, FxGraphLang::Placeholder(name.to_string()))
}

fn find_kw_arg<'a, T: FromStr>(node: &'a Node, key: &str) -> Result<Option<T>, Error> {
    let x = node
        .kwargs()
        .iter()
        .flatten()
        .find(|x| x.key() == Some(key))
        .and_then(|x| x.value())
        .and_then(|x| Some(x.parse::<T>()));

    match x {
        Some(Ok(v)) => return Ok(Some(v)),
        Some(Err(_)) => {
            return Err(Error::GraphNodeInvalidArgs(format!("{:?} {}", node, key)));
        }
        None => return Ok(None),
    }
}

fn handle_placeholder(fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    let target = node
        .target()
        .ok_or_else(|| Error::NoGraphNodeTarget(format!("{node:?}")))?;

    fxgraph.add_operation(
        node.name()
            .ok_or_else(|| Error::NoGraphNodeName(format!("{node:?}")))?,
        FxGraphLang::Placeholder(target.to_string()),
    );

    Ok(())
}

fn call_function(fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
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

fn mul(fxgraph: &mut FXGraph, node: &Node, name: &str, args: &[&str]) -> Result<(), Error> {
    if args.len() != 2 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let arg1 = find_or_create(fxgraph, args[0]);
    let arg2 = find_or_create(fxgraph, args[1]);

    fxgraph.add_operation(name, FxGraphLang::Mul([arg1, arg2]));

    Ok(())
}

fn rsqrt(fxgraph: &mut FXGraph, node: &Node, name: &str, args: &[&str]) -> Result<(), Error> {
    if args.len() != 1 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let arg1 = find_or_create(fxgraph, args[0]);
    fxgraph.add_operation(name, FxGraphLang::Rsqrt([arg1]));

    Ok(())
}

fn linear(fxgraph: &mut FXGraph, node: &Node, name: &str, args: &[&str]) -> Result<(), Error> {
    if args.len() != 2 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let arg1 = find_or_create(fxgraph, args[0]);
    let arg2 = find_or_create(fxgraph, args[1]);

    fxgraph.add_operation(name, FxGraphLang::Linear([arg1, arg2]));

    Ok(())
}

fn neg(fxgraph: &mut FXGraph, node: &Node, name: &str, args: &[&str]) -> Result<(), Error> {
    if args.len() != 1 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let arg1 = find_or_create(fxgraph, args[0]);
    fxgraph.add_operation(name, FxGraphLang::Neg(arg1));

    Ok(())
}

fn sym_sum(fxgraph: &mut FXGraph, node: &Node, name: &str, args: &[&str]) -> Result<(), Error> {
    if args.len() != 1 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    if args[0].starts_with("[") && args[0].ends_with("]") {
        let args = args[0][1..args[0].len() - 1]
            .split(",")
            .map(|x| find_or_create(fxgraph, x))
            .collect::<Vec<_>>();
        fxgraph.add_operation(name, FxGraphLang::SymSum(args));
    } else {
        return Err(Error::GraphNodeInvalidArgs(format!("sym_sum - {:?}", node)));
    }

    Ok(())
}

fn scaled_dot_product_attention(
    fxgraph: &mut FXGraph,
    node: &Node,
    name: &str,
    args: &[&str],
) -> Result<(), Error> {
    if args.len() != 3 {
        return Err(Error::GraphNodeMissingArgs(format!(
            "arg len  {} - {:?}",
            args.len(),
            node
        )));
    }

    let arg1 = find_or_create(fxgraph, args[0]);
    let arg2 = find_or_create(fxgraph, args[1]);
    let arg3 = find_or_create(fxgraph, args[2]);

    let attn_mask = find_or_create(
        fxgraph,
        &find_kw_arg::<String>(node, "attn_mask")?.ok_or(Error::GraphNodeMissingArgs(format!(
            "attn_mask - {:?}",
            node
        )))?,
    );

    let dropout_p =
        fxgraph.add_operation(
            &fxgraph.unique_name(),
            const_f32(find_kw_arg::<f32>(node, "dropout_p")?.ok_or(
                Error::GraphNodeMissingArgs(format!("dropout_p - {:?}", node)),
            )?),
        );

    let scale = find_or_create(
        fxgraph,
        &find_kw_arg::<String>(node, "scale")?
            .ok_or(Error::GraphNodeMissingArgs(format!("scale - {:?}", node)))?,
    );

    let is_causal = fxgraph.add_operation(
        &fxgraph.unique_name(),
        const_bool(&find_kw_arg::<String>(node, "is_causal")?.ok_or(
            Error::GraphNodeMissingArgs(format!("is_causal - {:?}", node)),
        )?),
    );

    fxgraph.add_operation(
        name,
        FxGraphLang::ScaledDotProductAttention([
            arg1, arg2, arg3, attn_mask, dropout_p, scale, is_causal,
        ]),
    );

    Ok(())
}

fn log_api_usage_once(
    _fxgraph: &mut FXGraph,
    _node: &Node,
    _name: &str,
    _args: &[&str],
) -> Result<(), Error> {
    // no-op
    Ok(())
}

fn getitem(fxgraph: &mut FXGraph, node: &Node, name: &str, args: &[&str]) -> Result<(), Error> {
    if args.len() != 2 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let arg1 = find_or_create(fxgraph, args[0]);
    let arg2 = fxgraph.add_operation(&fxgraph.unique_name(), const_string(args[1]));

    fxgraph.add_operation(name, FxGraphLang::GetItem([arg1, arg2]));

    Ok(())
}

fn lazy_load_decompositions(
    fxgraph: &mut FXGraph,
    _node: &Node,
    name: &str,
    _args: &[&str],
) -> Result<(), Error> {
    fxgraph.add_operation(name, FxGraphLang::LazyLoadDecompositions([]));
    Ok(())
}

fn silu(fxgraph: &mut FXGraph, node: &Node, name: &str, args: &[&str]) -> Result<(), Error> {
    if args.len() != 1 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let arg1 = find_or_create(fxgraph, args[0]);
    let inplace =
        const_bool(&find_kw_arg::<String>(node, "inplace")?.unwrap_or("false".to_string()));
    let arg2 = fxgraph.add_operation(&fxgraph.unique_name(), inplace);

    fxgraph.add_operation(name, FxGraphLang::Silu([arg1, arg2]));

    Ok(())
}

fn aten_index(fxgraph: &mut FXGraph, node: &Node, name: &str, args: &[&str]) -> Result<(), Error> {
    if args.len() != 2 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let arg1 = find_or_create(fxgraph, args[0]);
    let arg2 = find_or_create(fxgraph, args[1]);

    fxgraph.add_operation(name, FxGraphLang::AtenIndex([arg1, arg2]));

    Ok(())
}

fn add(fxgraph: &mut FXGraph, node: &Node, name: &str, args: &[&str]) -> Result<(), Error> {
    if args.len() != 2 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let arg1 = find_or_create(fxgraph, args[0]);
    let arg2 = find_or_create(fxgraph, args[1]);

    fxgraph.add_operation(name, FxGraphLang::Add([arg1, arg2]));

    Ok(())
}

fn iadd(fxgraph: &mut FXGraph, node: &Node, name: &str, args: &[&str]) -> Result<(), Error> {
    if args.len() != 2 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let arg1 = find_or_create(fxgraph, args[0]);
    let arg2 = find_or_create(fxgraph, args[1]);
    fxgraph.add_operation(name, FxGraphLang::IAdd([arg1, arg2]));
    Ok(())
}

fn arange(fxgraph: &mut FXGraph, node: &Node, name: &str, args: &[&str]) -> Result<(), Error> {
    let start;
    let end;
    let step;

    match args.len() {
        1 => {
            start = fxgraph.add_operation(&fxgraph.unique_name(), const_i64(0));
            end = find_or_create(fxgraph, args[0]);
            step = fxgraph.add_operation(&fxgraph.unique_name(), const_i64(1));
        }
        2 => {
            start = find_or_create(fxgraph, args[0]);
            end = find_or_create(fxgraph, args[1]);
            step = fxgraph.add_operation(&fxgraph.unique_name(), const_i64(1));
        }
        3 => {
            start = find_or_create(fxgraph, args[0]);
            end = find_or_create(fxgraph, args[1]);
            step = find_or_create(fxgraph, args[2]);
        }
        _ => {
            return Err(Error::GraphNodeMissingArgs(format!("args len: {node:?}")));
        }
    }

    fxgraph.add_operation(name, FxGraphLang::Arange([start, end, step]));

    Ok(())
}

fn embedding(fxgraph: &mut FXGraph, node: &Node, name: &str, args: &[&str]) -> Result<(), Error> {
    if args.len() != 5 {
        return Err(Error::GraphNodeMissingArgs(format!("args len: {node:?}")));
    }

    let input_ids = find_or_create(fxgraph, args[0]);
    let weight = find_or_create(fxgraph, args[1]);

    // AXM TODO: What is this arg?
    let item = find_or_create(fxgraph, args[2]);

    let arg3 = find_or_create(fxgraph, args[3]);
    let arg4 = find_or_create(fxgraph, args[4]);

    fxgraph.add_operation(
        name,
        FxGraphLang::Embedding([input_ids, weight, item, arg3, arg4]),
    );

    Ok(())
}

fn add_batch_dim(
    fxgraph: &mut FXGraph,
    node: &Node,
    name: &str,
    args: &[&str],
) -> Result<(), Error> {
    if args.len() != 3 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let arg1 = find_or_create(fxgraph, args[0]);
    let arg2 = find_or_create(fxgraph, args[1]);
    let arg3 = find_or_create(fxgraph, args[2]);
    fxgraph.add_operation(name, FxGraphLang::AddBatchDim([arg1, arg2, arg3]));

    Ok(())
}

fn remove_batch_dim(
    fxgraph: &mut FXGraph,
    node: &Node,
    name: &str,
    args: &[&str],
) -> Result<(), Error> {
    if args.len() != 4 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let arg1 = find_or_create(fxgraph, args[0]);
    let arg2 = find_or_create(fxgraph, args[1]);
    let arg3 = find_or_create(fxgraph, args[2]);
    let arg4 = find_or_create(fxgraph, args[3]);
    fxgraph.add_operation(name, FxGraphLang::RemoveBatchDim([arg1, arg2, arg3, arg4]));

    Ok(())
}

fn vmap_decrement_nesting(
    fxgraph: &mut FXGraph,
    node: &Node,
    name: &str,
    args: &[&str],
) -> Result<(), Error> {
    if !args.is_empty() {
        return Err(Error::GraphNodeInvalidArgs(format!("{:?}", node)));
    }

    fxgraph.add_operation(name, FxGraphLang::VmapDecrementNesting([]));

    Ok(())
}

fn enter_autocast(
    fxgraph: &mut FXGraph,
    node: &Node,
    name: &str,
    args: &[&str],
) -> Result<(), Error> {
    if args.len() != 2 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let arg1 = fxgraph.add_operation(name, const_string(args[0]));
    let arg2 = find_or_create(fxgraph, args[1]);
    fxgraph.add_operation(name, FxGraphLang::EnterAutocast([arg1, arg2]));

    Ok(())
}

fn exit_autocast(
    fxgraph: &mut FXGraph,
    node: &Node,
    name: &str,
    args: &[&str],
) -> Result<(), Error> {
    if args.len() != 1 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let arg1 = fxgraph
        .get_node(args[0])
        .ok_or_else(|| Error::GraphNodeNotFound(format!("{:?}", node)))?;

    fxgraph.add_operation(name, FxGraphLang::ExitAutocast([arg1]));

    Ok(())
}

fn matmul(fxgraph: &mut FXGraph, node: &Node, name: &str, args: &[&str]) -> Result<(), Error> {
    if args.len() != 2 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let arg1 = find_or_create(fxgraph, args[0]);
    let arg2 = find_or_create(fxgraph, args[1]);

    fxgraph.add_operation(name, FxGraphLang::MatMul([arg1, arg2]));

    Ok(())
}

fn vmap_increment_nesting(
    fxgraph: &mut FXGraph,
    node: &Node,
    name: &str,
    args: &[&str],
) -> Result<(), Error> {
    if args.len() != 2 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let arg1 = find_or_create(fxgraph, args[0]);
    let arg2 = find_or_create(fxgraph, args[1]);
    fxgraph.add_operation(name, FxGraphLang::VmapIncrementNesting([arg1, arg2]));

    Ok(())
}

fn cat(fxgraph: &mut FXGraph, node: &Node, name: &str, args: &[&str]) -> Result<(), Error> {
    if args.len() != 1 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let arg1 = find_or_create(fxgraph, args[0]);

    let dim = find_kw_arg::<i64>(node, "dim")?.unwrap_or(0);
    let dim = fxgraph.add_operation(&fxgraph.unique_name(), const_i64(dim));
    let arg2 = fxgraph.add_operation(&fxgraph.unique_name(), const_kv("dim", dim));

    fxgraph.add_operation(name, FxGraphLang::Cat([arg1, arg2]));

    Ok(())
}

fn call_method(_fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    let _target = node
        .target()
        .ok_or_else(|| Error::NoGraphNodeTarget(format!("{node:?}")))?;
    let _op = node.op();
    let _args = node.args();

    Ok(())
}

fn handle_output(fxgraph: &mut FXGraph, node: &Node) -> Result<(), Error> {
    let args = node
        .args()
        .ok_or_else(|| Error::NoGraphNodeArgs(format!("{node:?}")))?
        .iter()
        .collect::<Vec<_>>();
    if args.len() != 1 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    if args[0].starts_with("(") && args[0].ends_with(")") {
        let args = args[0][1..args[0].len() - 1]
            .split(",")
            .map(|x| find_or_create(fxgraph, x))
            .collect::<Vec<_>>();
        let name = node
            .name()
            .ok_or_else(|| Error::NoGraphNodeName(format!("{node:?}")))?;
        fxgraph.add_operation(name, FxGraphLang::Output(args));
    } else {
        return Err(Error::GraphNodeInvalidArgs(format!("output - {:?}", node)));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::graph::deserialize_graph;

    use super::*;

    const CARGO_MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");

    #[test]
    fn test_qwen3_conversion() -> Result<(), Error> {
        let qwen_files = ["qwen3_1.bin.gz", "qwen3_2.bin.gz", "qwen3_3.bin.gz"];

        for filename in qwen_files {
            let file_path = format!("{CARGO_MANIFEST_DIR}/tests/data/{filename}");

            // Read and decompress gzip file
            let file = std::fs::File::open(&file_path).unwrap();
            let mut decoder = flate2::read::GzDecoder::new(file);
            let mut buffer = Vec::new();
            std::io::Read::read_to_end(&mut decoder, &mut buffer).unwrap();

            // Deserialize graph
            let graph = deserialize_graph(&buffer)?;

            // Convert to FXGraph
            let fxgraph = FXGraph::try_from(graph)?;
            println!("FXGraph: #nodes {}", fxgraph.node_map.len());
        }

        Ok(())
    }
}
