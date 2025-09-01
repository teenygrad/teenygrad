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

use crate::{error::Error, torch::Output};

pub fn output<'a>(fxgraph: &mut FXGraph, output: &Output<'a>) -> Result<(), Error> {
    let name = output
        .name()
        .ok_or_else(|| Error::NoGraphNodeName(format!("{output:?}")))?;

    let output = fxgraph.add_operation(name, FxGraphLang::Output(name.to_string()));
    fxgraph.outputs.push(output);

    Ok(())
}
