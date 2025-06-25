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

use teeny_core::TeenyModule;

use crate::{error::TeenyHFError, transformer::config::model_config::HiddenAct};

#[derive(Default)]
pub struct Silu;

impl TeenyModule for Silu {
    type Err = TeenyHFError;

    fn forward(&self, _x: &[u32]) -> std::result::Result<Vec<u32>, Self::Err> {
        todo!()
    }
}

pub fn get_activation(activation: HiddenAct) -> Box<dyn TeenyModule<Err = TeenyHFError>> {
    match activation {
        HiddenAct::Silu => Box::new(Silu),
    }
}
