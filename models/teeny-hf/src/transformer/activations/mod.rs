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

use teeny_core::{
    graph,
    graph::NodeRef,
    nn::{Module, module::NodeRefModule},
};

use crate::{error::Result, transformer::config::model_config::HiddenAct};

#[derive(Default)]
pub struct Silu;

impl<'data> Module<'data, NodeRef<'data>, NodeRef<'data>> for Silu {
    fn forward(&mut self, _x: NodeRef<'data>) -> Result<NodeRef<'data>> {
        todo!()
    }

    fn parameters(&self) -> Vec<graph::NodeRef<'data>> {
        todo!()
    }
}

pub fn get_activation<'data>(activation: &HiddenAct) -> Result<NodeRefModule<'data>> {
    match activation {
        HiddenAct::Silu => Ok(Box::new(Silu)),
    }
}
