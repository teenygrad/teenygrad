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

use teeny_core::{
    dtype::Dtype,
    graph,
    graph::NodeRef,
    nn::{Module, module::NodeRefModule},
};

use crate::{error::Result, transformer::config::model_config::HiddenAct};

#[derive(Default)]
pub struct Silu;

impl<'data, N: Dtype> Module<'data, N, NodeRef<'data, N>, NodeRef<'data, N>> for Silu {
    fn forward(&self, _x: NodeRef<'data, N>) -> Result<NodeRef<'data, N>> {
        todo!()
    }

    fn parameters(&self) -> Vec<graph::NodeRef<'data, N>> {
        todo!()
    }
}

pub fn get_activation<'data, N: Dtype>(activation: HiddenAct) -> Result<NodeRefModule<'data, N>> {
    match activation {
        HiddenAct::Silu => Ok(Box::new(Silu)),
    }
}
