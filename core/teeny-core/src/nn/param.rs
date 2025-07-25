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

use crate::{dtype::Dtype, graph::NodeRef, tensor::shape::DynamicShape};

pub trait Param<N: Dtype>: Sized + Clone {
    fn shape(&self) -> DynamicShape;

    fn grad(&self) -> Option<NodeRef<N>> {
        todo!()
    }

    fn zero_grad(&self) {
        todo!()
    }

    fn weights(&self) -> NodeRef<N> {
        todo!()
    }

    fn update(&mut self, _update: NodeRef<N>) {
        todo!()
    }
}
