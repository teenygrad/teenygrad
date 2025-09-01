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
use teeny_core::fxgraph::{FXGraph, lang::FxGraphLang, value::Value};

use crate::{
    error::Error,
    fxgraph::{keyvalue::into_keyvalue, value::into_value},
    torch::CallMethod,
};

type NodeHandler = fn(&mut FXGraph, &CallMethod, &str, Vec<Value>, Option<Id>) -> Result<(), Error>;
static METHODS: OnceLock<HashMap<String, NodeHandler>> = OnceLock::new();

fn get_methods() -> &'static HashMap<String, NodeHandler> {
    METHODS.get_or_init(|| {
        HashMap::from([
            ("item".to_string(), item as NodeHandler),
            ("to".to_string(), to as NodeHandler),
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
    let args: Vec<teeny_core::fxgraph::value::Value> = node
        .args()
        .unwrap_or_default()
        .iter()
        .map(|x| into_value(fxgraph, x))
        .collect::<Result<Vec<_>, Error>>()?;
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
    args: Vec<Value>,
    _kwargs: Option<Id>,
) -> Result<(), Error> {
    let args = args
        .into_iter()
        .map(|x| fxgraph.add_value(x))
        .collect::<Vec<_>>();

    fxgraph.add_operation(name, FxGraphLang::ItemMethod(args));
    Ok(())
}

fn to(
    fxgraph: &mut FXGraph,
    node: &CallMethod,
    name: &str,
    _args: Vec<Value>,
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
