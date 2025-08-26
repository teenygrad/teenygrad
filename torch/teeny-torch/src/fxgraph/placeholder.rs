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

use teeny_core::fxgraph::{
    FXGraph, lang::FxGraphLang, placeholder::PlaceholderValue, shape::SymInt,
};

use crate::{
    error::Error,
    fxgraph::find_or_create,
    graph::{Placeholder, PlaceholderWrapper},
};

pub fn handle_placeholder(fxgraph: &mut FXGraph, node: &PlaceholderWrapper) -> Result<(), Error> {
    let name = node
        .name()
        .ok_or_else(|| Error::NoGraphNodeName(format!("{node:?}")))?;

    let target = node
        .target()
        .ok_or_else(|| Error::NoGraphNodeTarget(format!("{node:?}")))?;

    let users = node
        .users()
        .unwrap_or_default()
        .iter()
        .map(|x| find_or_create(fxgraph, x))
        .collect::<Vec<_>>();

    let value = match node.value_type() {
        Placeholder::symint => handle_symint(node)?,
        Placeholder::tensor => todo!(),
        _ => unreachable!(), // required because value_type is an int and not an enum
    };

    let placeholder = teeny_core::fxgraph::placeholder::Placeholder {
        name: name.to_string(),
        target: target.to_string(),
        value,
        users,
    };

    let id = fxgraph.add_operation(
        &fxgraph.unique_name(),
        FxGraphLang::Placeholder(placeholder),
    );

    fxgraph.inputs.push(id);

    Ok(())
}

fn handle_symint(node: &PlaceholderWrapper) -> Result<PlaceholderValue, Error> {
    let value = node
        .value_as_symint()
        .ok_or_else(|| Error::InvalidBuffer(format!("{node:?}")))?
        .value()
        .ok_or_else(|| Error::InvalidBuffer(format!("{node:?}")))?;

    let sym_int = value.parse::<i64>().ok();
    if let Some(sym_int) = sym_int {
        Ok(PlaceholderValue::SymInt(SymInt::Int(sym_int)))
    } else {
        Ok(PlaceholderValue::SymInt(SymInt::Str(value.to_string())))
    }
}

fn handle_tensor(node: &PlaceholderWrapper) -> Result<PlaceholderValue, Error> {
    todo!()
}
