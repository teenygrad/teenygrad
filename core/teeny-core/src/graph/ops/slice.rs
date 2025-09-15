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

use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use crate::dtype::DtypeEnum;
use crate::error::Result;
use crate::graph::shape::DynamicShape;
use crate::graph::{NodeOp, NodeRef, ops::Op};

/// Enum representing different types of slice indices
#[derive(Debug, Clone)]
pub enum TensorIndex<'data> {
    Single(isize),
    Range(Range<isize>),
    RangeFrom(RangeFrom<isize>),
    RangeTo(RangeTo<isize>),
    RangeFull(RangeFull),
    RangeInclusive(RangeInclusive<isize>),
    RangeToInclusive(RangeToInclusive<isize>),
    NodeRef(NodeRef<'data>),
}

impl<'data> From<isize> for TensorIndex<'data> {
    fn from(value: isize) -> Self {
        TensorIndex::Single(value)
    }
}

impl<'data> From<Range<isize>> for TensorIndex<'data> {
    fn from(value: Range<isize>) -> Self {
        TensorIndex::Range(value)
    }
}

impl<'data> From<RangeFrom<isize>> for TensorIndex<'data> {
    fn from(value: RangeFrom<isize>) -> Self {
        TensorIndex::RangeFrom(value)
    }
}

impl<'data> From<RangeTo<isize>> for TensorIndex<'data> {
    fn from(value: RangeTo<isize>) -> Self {
        TensorIndex::RangeTo(value)
    }
}

impl<'data> From<RangeFull> for TensorIndex<'data> {
    fn from(value: RangeFull) -> Self {
        TensorIndex::RangeFull(value)
    }
}

impl<'data> From<RangeInclusive<isize>> for TensorIndex<'data> {
    fn from(value: RangeInclusive<isize>) -> Self {
        TensorIndex::RangeInclusive(value)
    }
}

impl<'data> From<RangeToInclusive<isize>> for TensorIndex<'data> {
    fn from(value: RangeToInclusive<isize>) -> Self {
        TensorIndex::RangeToInclusive(value)
    }
}

impl<'data> From<NodeRef<'data>> for TensorIndex<'data> {
    fn from(value: NodeRef<'data>) -> Self {
        TensorIndex::NodeRef(value)
    }
}

#[derive(Debug, Clone)]
pub struct SliceOp<'data> {
    node: NodeRef<'data>,
    indices: Vec<TensorIndex<'data>>,
}

impl<'data> SliceOp<'data> {
    pub fn new(node: &NodeRef<'data>, indices: Vec<TensorIndex<'data>>) -> Self {
        Self {
            node: node.clone(),
            indices,
        }
    }
}

impl<'data> Op for SliceOp<'data> {
    fn shape(&self) -> Result<DynamicShape> {
        println!("indices: {:?}", self.indices);
        todo!()
    }

    fn dtype(&self) -> DtypeEnum {
        self.node.dtype()
    }
}

impl<'data> From<SliceOp<'data>> for NodeRef<'data> {
    fn from(op: SliceOp<'data>) -> Self {
        NodeOp::Slice(op).into()
    }
}
