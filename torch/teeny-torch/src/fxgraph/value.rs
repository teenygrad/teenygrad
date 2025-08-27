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

use teeny_core::fxgraph::FXGraph;

use crate::{
    error::Error,
    torch::{Value, ValueWrapper},
};

pub fn handle_value<'a>(
    fxgraph: &mut FXGraph,
    value: ValueWrapper<'a>,
) -> Result<teeny_core::fxgraph::value::Value, Error> {
    let val_type = value.value_type();

    let res = match val_type {
        Value::valnode => handle_valnode(fxgraph, value)?,
        _ => todo!("ValueWrapper: {:?}", value),
    };

    Ok(res)
}

fn handle_valnode<'a>(
    fxgraph: &mut FXGraph,
    value: ValueWrapper<'a>,
) -> Result<teeny_core::fxgraph::value::Value, Error> {
    let node = value
        .value_as_valnode()
        .ok_or(Error::InvalidBuffer(format!("{value:?}")))?;

    let value = node
        .value()
        .ok_or(Error::InvalidBuffer(format!("{value:?}")))?
        .replace("\"", "");

    println!("value: {value}");

    let node = fxgraph
        .get_node(&value)
        .ok_or(Error::NoGraphNode(format!("{value:?}")))?;

    Ok(teeny_core::fxgraph::value::Value::Node(node))
}
