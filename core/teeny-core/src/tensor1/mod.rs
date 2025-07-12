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

use std::{any::Any, sync::Arc};

use crate::tensor1::shape::DynamicShape;

pub mod num;
pub mod ops;
pub mod shape;

pub use ops::*;

pub type DTensor<E> = Arc<dyn Tensor<DType = E, Shape = DynamicShape>>;

pub trait Tensor: Send + Sync + std::fmt::Debug + Any {
    type DType: num::Num;
    type Shape: shape::Shape;

    fn dtype(&self) -> Self::DType;

    fn shape(&self) -> Self::Shape;

    // Downcast
    fn as_any(&self) -> &dyn Any;

    fn add(&self, other: &DTensor<Self::DType>) -> DTensor<Self::DType>;
}
