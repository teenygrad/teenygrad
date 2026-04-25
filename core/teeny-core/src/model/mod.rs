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

use alloc::{boxed::Box, vec::Vec};

use crate::{device::Device, errors::Result, graph::Graph, utils::dag::Dag};

pub type NodeId = usize;

pub trait RuntimeContext<'a> {}

pub trait ExecutableOp {
    fn forward(&self, inputs: &[NodeId]) -> Result<NodeId>;
}

pub trait Lowering<'a> {
    fn lower(&self, graph: &Graph) -> Result<Dag<Box<&'static dyn ExecutableOp>>>;
}

pub trait Model<'a> {
    type Input;
    type Output;

    fn forward(&self, input: Self::Input) -> Result<Self::Output>;
}
