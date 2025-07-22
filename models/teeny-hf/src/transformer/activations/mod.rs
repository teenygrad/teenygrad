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

use teeny_core::{dtype::Dtype, graph::NodeRef, nn::Module};

use crate::{error::Error, error::Result, transformer::config::model_config::HiddenAct};

#[derive(Default)]
pub struct Silu;

impl<N: Dtype> Module<N> for Silu {
    type Err = Error;

    fn forward(&self, _x: NodeRef<N>) -> Result<NodeRef<N>> {
        todo!()
    }

    fn parameters(&self) -> Vec<teeny_core::graph::NodeRef<N>> {
        todo!()
    }
}

pub fn get_activation(activation: HiddenAct) -> Result<Box<dyn Module<f32, Err = Error>>> {
    match activation {
        HiddenAct::Silu => Ok(Box::new(Silu)),
    }
}
