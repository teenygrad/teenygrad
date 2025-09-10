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

use teeny_core::fxgraph::{FXGraph, lang::FxGraphLang, value::Value};

use crate::{error::Error, torch::PlaceholderWrapper};

pub fn handle_placeholder(
    fxgraph: &mut FXGraph,
    node: &PlaceholderWrapper,
    example_inputs: &[Value],
) -> Result<(), Error> {
    let name = node
        .name()
        .ok_or_else(|| Error::NoGraphNodeName(format!("{node:?}")))?;

    let target = node.target().map(|x| x.to_string());

    let users = vec![];

    println!("Example inputs: {:?}", example_inputs);

    let placeholder = teeny_core::fxgraph::placeholder::Placeholder {
        name: name.to_string(),
        target,
        users,
        example_input: example_inputs[fxgraph.inputs.len()].clone(),
    };

    println!("Added placeholder: {:?}", placeholder);
    let id = fxgraph.add_operation(name, FxGraphLang::Placeholder(placeholder));

    fxgraph.inputs.push(id);
    Ok(())
}
