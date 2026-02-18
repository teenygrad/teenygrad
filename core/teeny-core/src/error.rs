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

use crate::safetensors::SafeTensorsError;

pub type Result<T> = anyhow::Result<T>;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(std::io::Error),

    #[error("Lock error: {0}")]
    TryLockError(String),

    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    #[error("Invalid graph: {0}")]
    InvalidGraph(String),

    #[error("SafeTensors error: {0}")]
    SafeTensorsError(SafeTensorsError),

    #[error("Z3 error: {0}")]
    Z3(String),

    #[error("Invalid device: {0}")]
    InvalidDevice(String),

    #[error("Device type not found: {0}")]
    DeviceTypeNotFound(String),

    #[error("Mutex lock error: {0}")]
    MutexLockError(String),

    #[error("Invalid type conversion: {0}")]
    InvalidTypeConversion(String),

    #[error("Invalid tensor broadcast: {0}")]
    InvalidTensorBroadcast(String),
}
