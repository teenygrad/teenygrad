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

use teeny_driver::error::DriverError;
use teeny_driver::{device::Device, driver::CUDA_DRIVER_ID, driver_manager::DriverManager};

use crate::error::Result;
use crate::error::RuntimeError;

pub mod device;
pub mod error;

#[cfg(feature = "cpu")]
extern crate teeny_cpu;

#[cfg(feature = "cuda")]
extern crate teeny_cuda;

pub fn init() -> Result<()> {
    init_drivers()?;
    Ok(())
}

pub fn get_cuda_devices() -> Result<Vec<Arc<Mutex<dyn Device>>>> {
    let driver = DriverManager::driver(CUDA_DRIVER_ID)
        .map_err(RuntimeError::DriverError)?
        .ok_or(RuntimeError::DriverError(DriverError::NotFound(
            CUDA_DRIVER_ID.to_string(),
        )))?;

    let devices = driver
        .lock()
        .unwrap()
        .devices()
        .map_err(RuntimeError::DriverError)?;
    Ok(devices)
}

fn init_drivers() -> Result<()> {
    let drivers =
        DriverManager::drivers().map_err(|e| RuntimeError::FailedToGetDrivers(e.to_string()))?;

    for driver in drivers {
        driver
            .lock()
            .unwrap()
            .init()
            .map_err(RuntimeError::DriverError)?;
    }

    Ok(())
}
