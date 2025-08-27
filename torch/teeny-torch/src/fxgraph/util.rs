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

use std::str::FromStr;

use egg::Id;
use teeny_core::fxgraph::{
    FXGraph,
    lang::{const_bool, const_f32, const_i64, const_string},
};

use crate::{error::Error, torch::Node};

pub fn find_or_create(fxgraph: &mut FXGraph, name: &str) -> Id {
    let node = fxgraph.get_node(name);
    if let Some(id) = node {
        return id;
    }

    let v = name.parse::<i64>();
    if let Ok(v) = v {
        return fxgraph.add_operation(&fxgraph.unique_name(), const_i64(v));
    }

    let v = name.parse::<f32>();
    if let Ok(v) = v {
        return fxgraph.add_operation(&fxgraph.unique_name(), const_f32(v));
    }

    let v = name.to_lowercase().parse::<bool>();
    if let Ok(v) = v {
        return fxgraph.add_operation(&fxgraph.unique_name(), const_bool(&v.to_string()));
    }

    if name.starts_with("(") && name.ends_with(")") {
        todo!("tuple: {:?}", name);
    }

    fxgraph.add_operation(&fxgraph.unique_name(), const_string(name))
}

pub fn find_kw_arg<T: FromStr>(_node: &Node, _key: &str) -> Result<Option<T>, Error> {
    todo!()
    // let x = node
    //     .kwargs()
    //     .iter()
    //     .flatten()
    //     .find(|x| x.key() == Some(key))
    //     .and_then(|x| x.value())
    //     .map(|x| x.parse::<T>());

    // match x {
    //     Some(Ok(v)) => Ok(Some(v)),
    //     Some(Err(_)) => Err(Error::GraphNodeInvalidArgs(format!("{:?} {}", node, key))),
    //     None => Ok(None),
    // }
}
