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
