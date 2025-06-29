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

use derive_builder::Builder;

#[derive(Builder, Debug, Clone)]
pub struct Adam {
    #[builder(default = "0.001")]
    pub lr: f32,

    #[builder(default = "0.9")]
    pub beta1: f32,

    #[builder(default = "0.999")]
    pub beta2: f32,

    #[builder(default = "1e-8")]
    pub eps: f32,
}

impl Adam {
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
        }
    }

    pub fn zero_grad(&mut self) {
        todo!()
    }

    pub fn update(&mut self) {
        todo!()
    }
}
