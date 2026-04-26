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

use std::{collections::HashMap, marker::PhantomData};

use anyhow::anyhow;
use teeny_core::{
    graph::{DtypeRepr, Shape},
    model::Model,
    utils::dag::Dag,
};

use crate::{errors::Result, device::CudaDevice};

pub type NodeId = usize;

/// Runtime tensor reference used by the model launcher.
///
/// For now this is intentionally minimal: kernels ultimately receive raw pointer
/// arguments, and this handle is enough to wire a compiled graph end-to-end.
#[derive(Copy, Clone, Debug)]
pub struct TensorRef {
    ptr: *mut core::ffi::c_void,
}

impl TensorRef {
    pub fn new(ptr: *mut core::ffi::c_void) -> Self {
        Self { ptr }
    }

    pub fn as_mut_ptr(self) -> *mut core::ffi::c_void {
        self.ptr
    }
}

#[derive(Default)]
pub struct RuntimeCtx {
    tensors: HashMap<NodeId, TensorRef>,
}

impl RuntimeCtx {
    pub fn new(inputs: HashMap<NodeId, TensorRef>) -> Self {
        Self { tensors: inputs }
    }

    pub fn get(&self, id: NodeId) -> Result<TensorRef> {
        self.tensors
            .get(&id)
            .copied()
            .ok_or_else(|| anyhow!("missing tensor for node id {id}"))
    }

    pub fn insert(&mut self, id: NodeId, tensor: TensorRef) {
        self.tensors.insert(id, tensor);
    }

    pub fn take(&mut self, id: NodeId) -> Result<TensorRef> {
        self.tensors
            .remove(&id)
            .ok_or_else(|| anyhow!("missing output tensor for node id {id}"))
    }
}

pub struct LaunchRequest {
    pub inputs: HashMap<NodeId, TensorRef>,
}

pub struct LaunchResult {
    pub output: TensorRef,
}

/// A compiled DAG node: the kernel has been compiled to a PTX object file on
/// disk. The runtime reads this file and loads it via `CudaProgram::try_from_ptx`
/// when the model is executed.
pub struct CompiledNode {
    /// Path to the compiled PTX `.o` file produced by `LlvmCompiler`. Empty for
    /// `Input` placeholder nodes, which have no kernel.
    pub ptx_path: String,
    pub entry_point: String,
    pub output_shape: Shape,
    pub output_dtype: DtypeRepr,
}

pub struct CudaModel<'a> {
    pub dag: Dag<CompiledNode>,
    _marker: PhantomData<&'a ()>,
}

impl<'a> Model<'a> for CudaModel<'a> {
    type Input = LaunchRequest;
    type Output = LaunchResult;

    fn forward(&self, _input: Self::Input) -> Result<Self::Output> {
        todo!();
    }
}

impl<'a> CudaModel<'a> {
    pub fn new(dag: Dag<CompiledNode>) -> Result<Self> {
        Ok(Self {
            dag,
            _marker: PhantomData,
        })
    }

    pub fn forward(
        &self,
        _device: &CudaDevice<'a>,
        _request: LaunchRequest,
    ) -> Result<LaunchResult> {
        todo!();
    }
}
