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
use teeny_core::{graph::Graph, model::Model};

use crate::{device::CudaDevice, errors::Result};

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

/// Type-erased executable operator in the model chain.
///
/// Concrete implementations can keep kernel-specific generic types internally
/// while exposing a uniform runtime API for model execution.
pub trait ExecutableOp<'a> {
    fn forward(&self, device: &CudaDevice<'a>, rt: &mut RuntimeCtx) -> Result<()>;
    fn backward(&self, device: &CudaDevice<'a>, rt: &mut RuntimeCtx) -> Result<()>;
}

pub struct LaunchRequest {
    pub inputs: HashMap<NodeId, TensorRef>,
}

pub struct LaunchResult {
    pub output: TensorRef,
}

pub struct CudaModel<'a> {
    ops: Vec<Box<dyn ExecutableOp<'a> + 'a>>,
    output_id: NodeId,
    _marker: PhantomData<&'a ()>,
}

impl<'a> Model<'a> for CudaModel<'a> {
    type Device = CudaDevice<'a>;
    type Input = LaunchRequest;
    type Output = LaunchResult;

    fn forward(&self, device: &Self::Device, input: Self::Input) -> Result<Self::Output> {
        self.forward(device, input)
    }
}

impl<'a> CudaModel<'a> {
    pub fn from_graph(_graph: &Graph, _device: &CudaDevice<'a>) -> Self {
        Self {
            ops: Vec::new(),
            output_id: 0,
            _marker: PhantomData,
        }
    }

    pub fn with_ops(mut self, ops: Vec<Box<dyn ExecutableOp<'a> + 'a>>) -> Self {
        self.ops = ops;
        self
    }

    pub fn with_output_id(mut self, output_id: NodeId) -> Self {
        self.output_id = output_id;
        self
    }

    pub fn forward(&self, device: &CudaDevice<'a>, request: LaunchRequest) -> Result<LaunchResult> {
        let mut rt = RuntimeCtx::new(request.inputs);
        for op in &self.ops {
            op.forward(device, &mut rt)?;
        }
        let output = rt.take(self.output_id)?;
        Ok(LaunchResult { output })
    }
}
