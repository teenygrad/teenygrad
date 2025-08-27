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

use teeny_core::fxgraph::{FXGraph, lang::FxGraphLang};

use crate::{error::Error, fxgraph::find_or_create, torch::PlaceholderWrapper};

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

    let placeholder = teeny_core::fxgraph::placeholder::Placeholder {
        name: name.to_string(),
        target: target.to_string(),
        users,
    };

    let id = fxgraph.add_operation(name, FxGraphLang::Placeholder(placeholder));

    fxgraph.inputs.push(id);

    Ok(())
}
