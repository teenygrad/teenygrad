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
