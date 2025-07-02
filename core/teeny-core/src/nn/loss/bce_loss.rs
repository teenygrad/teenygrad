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

use crate::{
    nn::loss::{Loss, LossFn},
    tensor::{Tensor, log, value::topsort_graph},
};

#[derive(Debug, Clone, Default)]
pub struct BCELoss {}

impl BCELoss {
    pub fn new() -> Self {
        Self {}
    }
}

impl LossFn for BCELoss {
    fn compute(&self, p: &Tensor, y: &Tensor) -> Loss {
        let loss = -(y * log(p.clone()) + (1.0 - y) * log(1.0 - p.clone()));

        Loss {
            params: topsort_graph(&p.value),
            loss,
        }
    }
}
