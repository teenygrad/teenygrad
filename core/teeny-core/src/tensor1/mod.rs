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

pub trait Num {}

pub trait Dimension {}

pub type DynTensor<E, S> = Arc<dyn Tensor<Elem = E, Shape = S>>;

pub trait Tensor: Send + Sync + std::fmt::Debug + Any {
    type Elem: Num;
    type Shape: Dimension;

    // Accessors
    fn shape(&self) -> Self::Shape;
    fn device(&self) -> Arc<dyn Device>;

    // Downcast
    fn as_any(&self) -> &dyn Any;

    fn to_device(&self, device: Arc<dyn Device>) -> DynTensor<Self::Elem, Self::Shape>;

    fn add(&self, other: &DynTensor<Self::Elem, Self::Shape>)
    -> DynTensor<Self::Elem, Self::Shape>;

    fn dot(&self, other: &DynTensor<Self::Elem, Self::Shape>)
    -> DynTensor<Self::Elem, Self::Shape>;
}
