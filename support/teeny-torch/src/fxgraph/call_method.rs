/*
 * Copyright (c) 2026 Teenygrad.
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

use std::{collections::HashMap, sync::OnceLock};

use egg::Id;
use teeny_core::fxgraph::{FXGraph, lang::FxGraphLang};

use crate::{
    error::Error,
    fxgraph::{keyvalue::into_keyvalue, value::into_value},
    torch::CallMethod,
};

type NodeHandler = fn(&mut FXGraph, &CallMethod, &str, Vec<Id>, Option<Id>) -> Result<(), Error>;
static METHODS: OnceLock<HashMap<String, NodeHandler>> = OnceLock::new();

fn get_methods() -> &'static HashMap<String, NodeHandler> {
    METHODS.get_or_init(|| {
        HashMap::from([
            ("item".to_string(), item as NodeHandler),
            ("to".to_string(), to as NodeHandler),
            ("new_ones".to_string(), new_ones as NodeHandler),
            ("le".to_string(), le as NodeHandler),
            ("__and__".to_string(), and as NodeHandler),
            ("float".to_string(), float as NodeHandler),
            ("expand".to_string(), expand as NodeHandler),
            ("transpose".to_string(), transpose as NodeHandler),
            ("cos".to_string(), cos as NodeHandler),
            ("sin".to_string(), sin as NodeHandler),
            ("pow".to_string(), pow as NodeHandler),
            ("mean".to_string(), mean as NodeHandler),
            ("view".to_string(), view as NodeHandler),
            ("unsqueeze".to_string(), unsqueeze as NodeHandler),
            ("reshape".to_string(), reshape as NodeHandler),
            ("contiguous".to_string(), contiguous as NodeHandler),
            ("numel".to_string(), numel as NodeHandler),
        ])
    })
}

pub fn call_method<'a>(fxgraph: &mut FXGraph, node: &CallMethod<'a>) -> Result<(), Error> {
    let name = node
        .name()
        .ok_or_else(|| Error::NoGraphNodeName(format!("{node:?}")))?;
    let target = node
        .target()
        .ok_or_else(|| Error::NoGraphNodeTarget(format!("{node:?}")))?;
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

    let methods = get_methods();
    if let Some(method) = methods.get(target) {
        method(fxgraph, node, name, args, kwargs)?;
    } else {
        unimplemented!("CallMethod: {:?}", node);
    }

    Ok(())
}

fn item(
    fxgraph: &mut FXGraph,
    _node: &CallMethod,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    fxgraph.add_operation(name, FxGraphLang::ItemMethod(args));
    Ok(())
}

fn to(
    fxgraph: &mut FXGraph,
    node: &CallMethod,
    name: &str,
    _args: Vec<Id>,
    kwargs: Option<Id>,
) -> Result<(), Error> {
    let args = node
        .args()
        .ok_or_else(|| Error::GraphNodeMissingArgs(format!("{node:?}")))?
        .into_iter()
        .map(|x| into_value(fxgraph, x))
        .collect::<Result<Vec<_>, Error>>()?;
    let mut args = vec![fxgraph.add_args(args)];

    if let Some(kwargs) = kwargs {
        args.push(kwargs);
    }

    // pytoch: torch.tensor.to(*args, **kwargs) - yuck!
    fxgraph.add_operation(name, FxGraphLang::To(args));

    Ok(())
}

fn new_ones(
    fxgraph: &mut FXGraph,
    _node: &CallMethod,
    name: &str,
    mut args: Vec<Id>,
    kwargs: Option<Id>,
) -> Result<(), Error> {
    if let Some(kwargs) = kwargs {
        args.push(kwargs);
    }

    fxgraph.add_operation(name, FxGraphLang::NewOnes(args));
    Ok(())
}

fn le(
    fxgraph: &mut FXGraph,
    _node: &CallMethod,
    name: &str,
    mut args: Vec<Id>,
    kwargs: Option<Id>,
) -> Result<(), Error> {
    if let Some(kwargs) = kwargs {
        args.push(kwargs);
    }

    fxgraph.add_operation(name, FxGraphLang::Le(args));
    Ok(())
}

fn and(
    fxgraph: &mut FXGraph,
    _node: &CallMethod,
    name: &str,
    mut args: Vec<Id>,
    kwargs: Option<Id>,
) -> Result<(), Error> {
    if let Some(kwargs) = kwargs {
        args.push(kwargs);
    }

    fxgraph.add_operation(name, FxGraphLang::And(args));
    Ok(())
}

fn float(
    fxgraph: &mut FXGraph,
    node: &CallMethod,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    if args.len() != 1 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let args = [args[0]];
    fxgraph.add_operation(name, FxGraphLang::Float(args));
    Ok(())
}

fn expand(
    fxgraph: &mut FXGraph,
    _node: &CallMethod,
    name: &str,
    mut args: Vec<Id>,
    kwargs: Option<Id>,
) -> Result<(), Error> {
    if let Some(kwargs) = kwargs {
        args.push(kwargs);
    }

    fxgraph.add_operation(name, FxGraphLang::Expand(args));
    Ok(())
}

fn transpose(
    fxgraph: &mut FXGraph,
    _node: &CallMethod,
    name: &str,
    mut args: Vec<Id>,
    kwargs: Option<Id>,
) -> Result<(), Error> {
    if let Some(kwargs) = kwargs {
        args.push(kwargs);
    }

    fxgraph.add_operation(name, FxGraphLang::Transpose(args));
    Ok(())
}

fn cos(
    fxgraph: &mut FXGraph,
    node: &CallMethod,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    if args.len() != 1 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let args = [args[0]];
    fxgraph.add_operation(name, FxGraphLang::Cos(args));
    Ok(())
}

fn sin(
    fxgraph: &mut FXGraph,
    node: &CallMethod,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    if args.len() != 1 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let args = [args[0]];
    fxgraph.add_operation(name, FxGraphLang::Sin(args));
    Ok(())
}

fn pow(
    fxgraph: &mut FXGraph,
    node: &CallMethod,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    if args.len() != 2 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let args = [args[0], args[1]];
    fxgraph.add_operation(name, FxGraphLang::Pow(args));
    Ok(())
}

fn mean(
    fxgraph: &mut FXGraph,
    _node: &CallMethod,
    name: &str,
    mut args: Vec<Id>,
    kwargs: Option<Id>,
) -> Result<(), Error> {
    if let Some(kwargs) = kwargs {
        args.push(kwargs);
    }

    fxgraph.add_operation(name, FxGraphLang::Mean(args));
    Ok(())
}

fn view(
    fxgraph: &mut FXGraph,
    node: &CallMethod,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    if args.len() != 2 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let args = [args[0], args[1]];
    fxgraph.add_operation(name, FxGraphLang::View(args));

    Ok(())
}

fn unsqueeze(
    fxgraph: &mut FXGraph,
    node: &CallMethod,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    if args.len() != 2 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    let args = [args[0], args[1]];
    fxgraph.add_operation(name, FxGraphLang::Unsqueeze(args));
    Ok(())
}

fn reshape(
    fxgraph: &mut FXGraph,
    _node: &CallMethod,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    fxgraph.add_operation(name, FxGraphLang::Reshape(args));
    Ok(())
}

fn contiguous(
    fxgraph: &mut FXGraph,
    node: &CallMethod,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    if args.len() != 1 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    fxgraph.add_operation(name, FxGraphLang::Contiguous(args[0]));
    Ok(())
}

fn numel(
    fxgraph: &mut FXGraph,
    node: &CallMethod,
    name: &str,
    args: Vec<Id>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    if args.len() != 1 {
        return Err(Error::GraphNodeMissingArgs(format!("{:?}", node)));
    }

    fxgraph.add_operation(name, FxGraphLang::Numel(args[0]));
    Ok(())
}
