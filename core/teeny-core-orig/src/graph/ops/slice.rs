/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
