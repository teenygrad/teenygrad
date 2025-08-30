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

use teeny_core::fxgraph::{FXGraph, lang::FxGraphLang, torch::item::Item, value::Value};

use crate::{
    error::Error,
    fxgraph::{util::find_kw_arg, value::into_value},
    torch::CallMethod,
};

type NodeHandler = fn(&mut FXGraph, &CallMethod, &str, Vec<Value>) -> Result<(), Error>;
static METHODS: OnceLock<HashMap<String, NodeHandler>> = OnceLock::new();

// Initialize the functions HashMap
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
        .ok_or_else(|| Error::NoGraphNodeArgs(format!("{node:?}")))?
        .iter()
        .map(|x| into_value(fxgraph, x))
        .collect::<Result<Vec<_>, Error>>()?;

    let methods = get_methods();
    if let Some(method) = methods.get(target) {
        method(fxgraph, node, name, args)?;
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
) -> Result<(), Error> {
    let item = Item {
        name: name.to_string(),
        args,
    };
    fxgraph.add_operation(name, FxGraphLang::Item(item));
    Ok(())
}

fn to(
    fxgraph: &mut FXGraph,
    node: &CallMethod,
    name: &str,
    _args: Vec<Value>,
) -> Result<(), Error> {
    let key_values = node
        .kwargs()
        .ok_or_else(|| Error::GraphNodeMissingArgs(format!("{node:?}")))?
        .iter()
        .collect::<Vec<_>>();
    let device = into_value(
        fxgraph,
        find_kw_arg(&key_values, "device")?
            .ok_or_else(|| Error::GraphNodeMissingArgs(format!("{node:?}")))?,
    )?;
    let dtype = into_value(
        fxgraph,
        find_kw_arg(&key_values, "dtype")?
            .ok_or_else(|| Error::GraphNodeMissingArgs(format!("{node:?}")))?,
    )?;

    let device = fxgraph.add_operation(name, FxGraphLang::Value(device));
    let dtype = fxgraph.add_operation(name, FxGraphLang::Value(dtype));
    fxgraph.add_operation(name, FxGraphLang::To([device, dtype]));

    Ok(())
}
