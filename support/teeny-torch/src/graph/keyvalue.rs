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

use crate::graph::value::into_value;
use crate::{error::Error, torch::KeyValue};
use teeny_fxgraph::fxgraph::FXGraph;
use teeny_fxgraph::fxgraph::keyvalue::KeyValue as CoreKeyValue;

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

    Ok(teeny_fxgraph::fxgraph::keyvalue::KeyValue::Kv(
        key.to_string(),
        value,
    ))
}
