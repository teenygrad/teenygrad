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

use std::fmt::{Display, Formatter};
use std::str::FromStr;

use egg::Id;

use crate::error::Error;
use crate::fxgraph::dtype::Dtype;
use crate::fxgraph::shape::{Shape, SymInt};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Tensor {
    dtype: Dtype,
    shape: Shape,
    stride: Vec<usize>,
    requires_grad: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PlaceholderValue {
    SymInt(SymInt),
    Tensor(Tensor),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Placeholder {
    pub name: String,
    pub target: String,
    pub value: PlaceholderValue,
    pub users: Vec<Id>,
}

impl FromStr for Placeholder {
    type Err = Error;

    fn from_str(_s: &str) -> core::result::Result<Self, Self::Err> {
        todo!()
    }
}

impl Display for Placeholder {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        format!("{:?}", self).fmt(f)
    }
}
