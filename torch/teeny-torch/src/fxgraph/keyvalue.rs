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

use crate::fxgraph::value::into_value;
use crate::torch::Node;
use crate::{error::Error, torch::KeyValue};
use teeny_core::fxgraph::FXGraph;
use teeny_core::fxgraph::keyvalue::KeyValue as CoreKeyValue;
use teeny_core::fxgraph::keyvalue::KeyValueList as CoreKeyValueList;

pub fn into_keyvalue<'a>(
    fxgraph: &mut FXGraph,
    value: KeyValue<'a>,
) -> Result<CoreKeyValue, Error> {
    let key = value
        .key()
        .ok_or_else(|| Error::GraphNodeMissingArgs(format!("{value:?}")))?;
    let value = value
        .value()
        .ok_or_else(|| Error::GraphNodeMissingArgs(format!("{value:?}")))?;
    let value = into_value(fxgraph, value)?;

    Ok(teeny_core::fxgraph::keyvalue::KeyValue::Kv(
        key.to_string(),
        fxgraph.add_value(value),
    ))
}
