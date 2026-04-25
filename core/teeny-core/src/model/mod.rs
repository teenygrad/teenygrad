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

use alloc::boxed::Box;

use crate::{errors::Result, graph::{DtypeRepr, Graph, Shape}, utils::dag::Dag};

pub type NodeId = usize;

pub trait RuntimeContext<'a> {}

/// An op that has been lowered to a compilable kernel representation.
///
/// Holds enough information for a caller (who has access to `teeny-compiler`)
/// to compile the kernel for a given target. Dispatch/execution is deferred.
pub trait ExecutableOp {
    fn kernel_source(&self) -> &str;
    fn kernel_entry_point(&self) -> &str;
    fn output_shape(&self) -> &Shape;
    fn output_dtype(&self) -> DtypeRepr;
}

pub trait Lowering<'a> {
    fn lower(&self, graph: &Graph) -> Result<Dag<Box<dyn ExecutableOp>>>;
}

pub trait Model<'a> {
    type Input;
    type Output;

    fn forward(&self, input: Self::Input) -> Result<Self::Output>;
}
