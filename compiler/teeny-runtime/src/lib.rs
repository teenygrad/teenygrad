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
        return Err(Error::NoDevicesAvailable);
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
