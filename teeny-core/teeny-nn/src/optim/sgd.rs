/*
 * Copyright (C) 2025 SpinorML Ltd.
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

use crate::model::Parameters;

use super::{Optimizer, error::OptimizerError};

pub struct SGD<'a, T> {
    pub lr: T,
    pub params: &'a dyn Parameters,
}

impl<'a, T> SGD<'a, T> {
    pub fn new(lr: T, params: &'a dyn Parameters) -> Self {
        Self { lr, params }
    }
}

impl<T> Optimizer<T> for SGD<'_, T> {
    fn step(&self) {
        todo!()
    }

    fn zero_grad(&mut self) {
        todo!()
    }
}

pub struct SGDBuilder<'a, T> {
    lr: Option<T>,
    params: Option<&'a dyn Parameters>,
}

impl<'a, T> SGDBuilder<'a, T> {
    pub fn lr(mut self, lr: T) -> Self {
        self.lr = Some(lr);
        self
    }

    pub fn parameters(mut self, params: &'a dyn Parameters) -> Self {
        self.params = Some(params);
        self
    }

    pub fn new() -> Self {
        Self {
            lr: None,
            params: None,
        }
    }
}

impl<T> Default for SGDBuilder<'_, T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> SGDBuilder<'a, f32> {
    pub fn build(self) -> Result<SGD<'a, f32>, OptimizerError> {
        if self.params.is_none() {
            return Err(OptimizerError::NoParameters());
        }

        Ok(SGD::new(self.lr.unwrap_or(0.01), self.params.unwrap()))
    }
}
