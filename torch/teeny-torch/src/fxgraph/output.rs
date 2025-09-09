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

use teeny_core::fxgraph::{FXGraph, keyvalue::KeyValueList, lang::FxGraphLang, value::Value};

use crate::{
    error::Error,
    fxgraph::{keyvalue::into_keyvalue, value::into_value},
    torch::Output,
};

pub fn output<'a>(fxgraph: &mut FXGraph, output: &Output<'a>) -> Result<(), Error> {
    let name = output
        .name()
        .ok_or_else(|| Error::NoGraphNodeName(format!("{output:?}")))?;

    let args = output
        .args()
        .unwrap_or_default()
        .iter()
        .map(|arg| into_value(fxgraph, arg))
        .collect::<Result<Vec<_>, Error>>()?
        .into_iter()
        .map(Box::new)
        .collect::<Vec<_>>();

    let kwargs = KeyValueList::new(
        output
            .kwargs()
            .unwrap_or_default()
            .iter()
            .map(|kv| into_keyvalue(fxgraph, kv))
            .collect::<Result<Vec<_>, Error>>()?,
    );

    let args = fxgraph.add_value(Value::List(args));
    let kwargs = fxgraph.add_operation(&fxgraph.unique_name(), FxGraphLang::KwArgs(kwargs));

    let output = fxgraph.add_operation(name, FxGraphLang::Output([args, kwargs]));
    fxgraph.outputs.push(output);

    Ok(())
}
