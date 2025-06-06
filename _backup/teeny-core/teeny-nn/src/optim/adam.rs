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

pub struct Adam<'a, T> {
    pub lr: T,
    pub beta1: T,
    pub beta2: T,
    pub eps: T,
    pub params: &'a dyn Parameters,
}

impl<'a, T> Adam<'a, T> {
    pub fn new(lr: T, beta1: T, beta2: T, eps: T, params: &'a dyn Parameters) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            params,
        }
    }
}

impl<T> Optimizer<T> for Adam<'_, T> {
    fn step(&self) {
        // TODO: Implement Adam step
    }

    fn zero_grad(&mut self) {
        todo!()
    }
}

pub struct AdamBuilder<'a, T> {
    lr: Option<T>,
    beta1: Option<T>,
    beta2: Option<T>,
    eps: Option<T>,
    params: Option<&'a dyn Parameters>,
}

impl<'a, T> AdamBuilder<'a, T> {
    pub fn lr(mut self, lr: T) -> Self {
        self.lr = Some(lr);
        self
    }

    pub fn beta1(mut self, beta1: T) -> Self {
        self.beta1 = Some(beta1);
        self
    }

    pub fn beta2(mut self, beta2: T) -> Self {
        self.beta2 = Some(beta2);
        self
    }

    pub fn eps(mut self, eps: T) -> Self {
        self.eps = Some(eps);
        self
    }

    pub fn parameters(mut self, params: &'a dyn Parameters) -> Self {
        self.params = Some(params);
        self
    }
}

impl<'a> AdamBuilder<'a, f32> {
    pub fn build(self) -> Result<Adam<'a, f32>, OptimizerError> {
        if self.params.is_none() {
            return Err(OptimizerError::NoParameters());
        }

        Ok(Adam::new(
            self.lr.unwrap_or(0.001),
            self.beta1.unwrap_or(0.9),
            self.beta2.unwrap_or(0.999),
            self.eps.unwrap_or(1e-8),
            self.params.unwrap(),
        ))
    }
}
