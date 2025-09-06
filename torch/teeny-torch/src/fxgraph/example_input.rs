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
