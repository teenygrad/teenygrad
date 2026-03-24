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

use teeny_fxgraph::fxgraph::FXGraph;

use crate::{
    error::Error,
    torch::{Value, ValueWrapper},
};

pub fn into_value<'a>(
    fxgraph: &mut FXGraph,
    value: ValueWrapper<'a>,
) -> Result<teeny_fxgraph::fxgraph::value::Value, Error> {
    let val_type = value.value_type();

    let res = match val_type {
        Value::valnode => valnode(fxgraph, value)?,
        Value::valnone | Value::NONE => valnone(fxgraph, value)?,
        Value::valint => valint(fxgraph, value)?,
        Value::valdevice => valdevice(fxgraph, value)?,
        Value::valdtype => valdtype(fxgraph, value)?,
        Value::valstr => valstr(fxgraph, value)?,
        Value::valtuple => valtuple(fxgraph, value)?,
        Value::vallist => vallist(fxgraph, value)?,
        Value::valslice => valslice(fxgraph, value)?,
        Value::valellipsis => valellipsis(fxgraph, value)?,
        _ => todo!("ValueWrapper: {:?}", value),
    };

    Ok(res)
}

fn valnode<'a>(
    fxgraph: &mut FXGraph,
    value: ValueWrapper<'a>,
) -> Result<teeny_fxgraph::fxgraph::value::Value, Error> {
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

    Ok(teeny_fxgraph::fxgraph::value::Value::Node(node))
}

fn valnone<'a>(
    _fxgraph: &mut FXGraph,
    _value: ValueWrapper<'a>,
) -> Result<teeny_fxgraph::fxgraph::value::Value, Error> {
    Ok(teeny_fxgraph::fxgraph::value::Value::None)
}

fn valint<'a>(
    _fxgraph: &mut FXGraph,
    value: ValueWrapper<'a>,
) -> Result<teeny_fxgraph::fxgraph::value::Value, Error> {
    let intval = value
        .value_as_valint()
        .ok_or(Error::InvalidBuffer(format!("{value:?}")))?;
    let value = intval.value();

    // AXM : what is the correct type for this?
    Ok(teeny_fxgraph::fxgraph::value::Value::Int64(value as i64))
}

fn valdevice<'a>(
    _fxgraph: &mut FXGraph,
    value: ValueWrapper<'a>,
) -> Result<teeny_fxgraph::fxgraph::value::Value, Error> {
    let deviceval = value
        .value_as_valdevice()
        .ok_or(Error::InvalidBuffer(format!("{value:?}")))?;
    let value = deviceval
        .value()
        .ok_or(Error::InvalidBuffer(format!("{value:?}")))?;

    // AXM : what is the correct type for this?
    Ok(teeny_fxgraph::fxgraph::value::Value::Device(
        value.to_string(),
    ))
}

fn valdtype<'a>(
    _fxgraph: &mut FXGraph,
    value: ValueWrapper<'a>,
) -> Result<teeny_fxgraph::fxgraph::value::Value, Error> {
    let dtypeval = value
        .value_as_valdtype()
        .ok_or(Error::InvalidBuffer(format!("{value:?}")))?;
    let value = dtypeval.value();

    Ok(teeny_fxgraph::fxgraph::value::Value::DType(
        value.try_into()?,
    ))
}

fn valstr<'a>(
    _fxgraph: &mut FXGraph,
    value: ValueWrapper<'a>,
) -> Result<teeny_fxgraph::fxgraph::value::Value, Error> {
    let strval = value
        .value_as_valstr()
        .ok_or(Error::InvalidBuffer(format!("{value:?}")))?;
    let value = strval
        .value()
        .ok_or(Error::InvalidBuffer(format!("{value:?}")))?;

    Ok(teeny_fxgraph::fxgraph::value::Value::String(
        value.to_string(),
    ))
}

fn valtuple<'a>(
    fxgraph: &mut FXGraph,
    value: ValueWrapper<'a>,
) -> Result<teeny_fxgraph::fxgraph::value::Value, Error> {
    let tupleval = value
        .value_as_valtuple()
        .ok_or(Error::InvalidBuffer(format!("{value:?}")))?;
    let values = tupleval
        .values()
        .ok_or(Error::InvalidBuffer(format!("{value:?}")))?
        .iter()
        .map(|x| into_value(fxgraph, x))
        .collect::<Result<Vec<teeny_fxgraph::fxgraph::value::Value>, Error>>()?
        .into_iter()
        .map(Box::new)
        .collect::<Vec<_>>();

    Ok(teeny_fxgraph::fxgraph::value::Value::Tuple(values))
}

fn vallist<'a>(
    fxgraph: &mut FXGraph,
    value: ValueWrapper<'a>,
) -> Result<teeny_fxgraph::fxgraph::value::Value, Error> {
    let listval = value
        .value_as_vallist()
        .ok_or(Error::InvalidBuffer(format!("{value:?}")))?;
    let values = listval
        .values()
        .ok_or(Error::InvalidBuffer(format!("{value:?}")))?
        .iter()
        .map(|x| into_value(fxgraph, x))
        .collect::<Result<Vec<teeny_fxgraph::fxgraph::value::Value>, Error>>()?
        .into_iter()
        .map(Box::new)
        .collect::<Vec<_>>();

    Ok(teeny_fxgraph::fxgraph::value::Value::List(values))
}

fn valslice<'a>(
    fxgraph: &mut FXGraph,
    value: ValueWrapper<'a>,
) -> Result<teeny_fxgraph::fxgraph::value::Value, Error> {
    let sliceval = value
        .value_as_valslice()
        .ok_or(Error::InvalidBuffer(format!("{value:?}")))?;
    let start = sliceval
        .start()
        .ok_or(Error::InvalidBuffer(format!("{value:?}")))?;
    let end = sliceval
        .end()
        .ok_or(Error::InvalidBuffer(format!("{value:?}")))?;
    let step = sliceval
        .step()
        .ok_or(Error::InvalidBuffer(format!("{value:?}")))?;

    Ok(teeny_fxgraph::fxgraph::value::Value::Slice(
        Box::new(into_value(fxgraph, start)?),
        Box::new(into_value(fxgraph, end)?),
        Box::new(into_value(fxgraph, step)?),
    ))
}

fn valellipsis<'a>(
    _fxgraph: &mut FXGraph,
    _value: ValueWrapper<'a>,
) -> Result<teeny_fxgraph::fxgraph::value::Value, Error> {
    Ok(teeny_fxgraph::fxgraph::value::Value::Ellipsis)
}
