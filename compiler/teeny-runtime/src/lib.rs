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

use std::sync::Arc;
use std::sync::Mutex;

use crate::device::Device;
use crate::error::Error;
use crate::error::Result;

use once_cell::sync::OnceCell;

#[cfg(feature = "compiler")]
pub mod compiler;

pub mod device;
pub mod error;

static CURRENT_DEVICE: OnceCell<Mutex<Option<Arc<Device>>>> = OnceCell::new();

pub fn current_device() -> Result<Option<Arc<Device>>> {
    let guard = CURRENT_DEVICE
        .get_or_init(|| Mutex::new(None))
        .try_lock()
        .map_err(|_| Error::TryLockError("Failed to lock device".to_string()))?;

    Ok((*guard).clone())
}

pub fn set_current_device(device: Arc<Device>) -> Result<()> {
    let mut guard = CURRENT_DEVICE
        .get_or_init(|| Mutex::new(Some(device.clone())))
        .try_lock()
        .map_err(|_| Error::TryLockError("Failed to set device".to_string()))?;
    *guard = Some(device.clone());
    Ok(())
}

pub fn init() -> Result<()> {
    use_fallback_device()?;

    let device = current_device()?;
    if let Some(device) = device {
        let id = device.id();
        let name = device.name();
        println!("Using device: {id} {name}");
    } else {
        return Err(Error::NoDevicesAvailable.into());
    }

    Ok(())
}

pub fn use_fallback_device() -> Result<()> {
    let devices = device::find_cpu_devices()?;
    if !devices.is_empty() {
        set_current_device(Arc::new(devices[0].clone()))?;
    }
    Ok(())
}

pub fn use_accelerator_with_fallback() -> Result<()> {
    let devices = device::find_cuda_devices()?;
    if !devices.is_empty() {
        set_current_device(Arc::new(devices[0].clone()))?;
    } else {
        use_fallback_device()?;
    }

    Ok(())
}
