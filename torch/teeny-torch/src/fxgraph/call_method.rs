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

use teeny_core::fxgraph::{FXGraph, lang::FxGraphLang, torch::item::Item};

use crate::{error::Error, fxgraph::value::value, torch::CallMethod};

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
        .map(|x| value(fxgraph, x))
        .collect::<Result<Vec<_>, Error>>()?;

    if target == "item" {
        let item = Item {
            name: name.to_string(),
            args,
        };
        fxgraph.add_operation(name, FxGraphLang::Item(item));
    } else {
        todo!("CallMethod: {:?}", node);
    }

    Ok(())
}
