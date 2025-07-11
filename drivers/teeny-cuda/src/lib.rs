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

use std::sync::{Arc, Mutex};

use ctor::ctor;
use teeny_driver::device::Device;
use teeny_driver::driver::{CUDA_DRIVER_ID, Driver};
use teeny_driver::driver_manager::DriverManager;
use teeny_driver::error::Result;

#[derive(Default)]
pub struct CudaDriver {}

impl CudaDriver {
    pub const fn new() -> Self {
        Self {}
    }
}

impl Driver for CudaDriver {
    fn init(&mut self) -> Result<()> {
        Ok(())
    }

    fn deinit(&mut self) -> Result<()> {
        // no-op
        Ok(())
    }

    fn id(&self) -> &str {
        CUDA_DRIVER_ID
    }

    fn name(&self) -> &str {
        "Teenygrad CUDA Driver v0.1.0"
    }

    fn devices(&self) -> Result<Vec<Arc<Mutex<dyn Device>>>> {
        Ok(vec![])
    }
}

#[ctor]
fn register_cuda() {
    DriverManager::register(Arc::new(Mutex::new(CudaDriver::new())));
}
