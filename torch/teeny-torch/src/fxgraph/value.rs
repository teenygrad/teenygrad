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

use egg::Id;
use teeny_core::fxgraph::FXGraph;

use crate::{
    error::Error,
    torch::{Value, ValueWrapper},
};

pub fn into_value<'a>(
    fxgraph: &mut FXGraph,
    value: ValueWrapper<'a>,
) -> Result<teeny_core::fxgraph::value::Value, Error> {
    let val_type = value.value_type();

    let res = match val_type {
        Value::valnode => valnode(fxgraph, value)?,
        Value::valnone => valnone(fxgraph, value)?,
        Value::valint => valint(fxgraph, value)?,
        Value::valdevice => valdevice(fxgraph, value)?,
        Value::valdtype => valdtype(fxgraph, value)?,
        _ => todo!("ValueWrapper: {:?}", value),
    };

    Ok(res)
}

pub fn node_value<'a>(fxgraph: &mut FXGraph, node: ValueWrapper<'a>) -> Result<Id, Error> {
    let node = valnode(fxgraph, node)?;

    match node {
        teeny_core::fxgraph::value::Value::Node(node) => Ok(node),
        _ => Err(Error::GraphNodeInvalidArgs(format!("{node:?}"))),
    }
}

fn valnode<'a>(
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

fn valnone<'a>(
    _fxgraph: &mut FXGraph,
    _value: ValueWrapper<'a>,
) -> Result<teeny_core::fxgraph::value::Value, Error> {
    Ok(teeny_core::fxgraph::value::Value::None)
}

fn valint<'a>(
    _fxgraph: &mut FXGraph,
    value: ValueWrapper<'a>,
) -> Result<teeny_core::fxgraph::value::Value, Error> {
    let intval = value
        .value_as_valint()
        .ok_or(Error::InvalidBuffer(format!("{value:?}")))?;
    let value = intval.value();

    // AXM : what is the correct type for this?
    Ok(teeny_core::fxgraph::value::Value::Int(value as i64))
}

fn valdevice<'a>(
    _fxgraph: &mut FXGraph,
    value: ValueWrapper<'a>,
) -> Result<teeny_core::fxgraph::value::Value, Error> {
    let deviceval = value
        .value_as_valdevice()
        .ok_or(Error::InvalidBuffer(format!("{value:?}")))?;
    let value = deviceval
        .value()
        .ok_or(Error::InvalidBuffer(format!("{value:?}")))?;

    // AXM : what is the correct type for this?
    Ok(teeny_core::fxgraph::value::Value::Device(value.to_string()))
}

fn valdtype<'a>(
    _fxgraph: &mut FXGraph,
    value: ValueWrapper<'a>,
) -> Result<teeny_core::fxgraph::value::Value, Error> {
    let dtypeval = value
        .value_as_valdtype()
        .ok_or(Error::InvalidBuffer(format!("{value:?}")))?;
    let value = dtypeval.value();

    Ok(teeny_core::fxgraph::value::Value::DType(value.try_into()?))
}
