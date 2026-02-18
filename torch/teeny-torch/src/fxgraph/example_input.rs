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

use teeny_core::fxgraph::{device::Device, tensor::Tensor, value::Value};

use crate::{
    error::Error,
    torch::{ExampleInput, ExampleInputWrapper},
};

pub fn into_example_input(example_input: ExampleInputWrapper) -> Result<Value, Error> {
    let value = match example_input.value_type() {
        ExampleInput::symint => handle_symint(example_input)?,
        ExampleInput::tensor => handle_tensor(example_input)?,
        _ => {
            return Err(Error::InvalidBuffer(format!(
                "Unsupported example input type: {:?}",
                example_input
            )));
        }
    };

    Ok(value)
}

fn handle_symint(example_input: ExampleInputWrapper) -> Result<Value, Error> {
    let symint = example_input
        .value_as_symint()
        .ok_or(Error::InvalidBuffer(format!(
            "Invalid symint example input: {:?}",
            example_input
        )))?;

    Ok(Value::SymInt(symint.try_into()?))
}

fn handle_tensor(example_input: ExampleInputWrapper) -> Result<Value, Error> {
    let tensor = example_input
        .value_as_tensor()
        .ok_or(Error::InvalidBuffer(format!(
            "Invalid tensor example input: {:?}",
            example_input
        )))?;

    let dtype = tensor.dtype().try_into()?;
    let shape = tensor
        .shape()
        .ok_or(Error::InvalidBuffer(format!(
            "Invalid tensor shape: {:?}",
            tensor
        )))?
        .try_into()?;
    let device = tensor.device().ok_or(Error::InvalidBuffer(format!(
        "Invalid tensor device: {:?}",
        tensor
    )))?;
    let stride = tensor
        .stride()
        .ok_or(Error::InvalidBuffer(format!(
            "Invalid tensor strides: {:?}",
            tensor
        )))?
        .into_iter()
        .collect::<Vec<_>>();
    let requires_grad = tensor.requires_grad();

    Ok(Value::Tensor(Tensor {
        dtype,
        shape,
        device: device
            .parse::<Device>()
            .map_err(|e| Error::ParsingDevice(e.to_string()))?,
        stride,
        requires_grad,
    }))
}
