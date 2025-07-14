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

use ndarray::IxDyn;
use teeny_core::tensor1::num;

pub mod device;
pub mod driver;
pub mod error;

// use teeny_core::device::Device;
// use teeny_driver::driver::{CPU_DRIVER_ID, Driver};
// use teeny_driver::driver_manager::DriverManager;
// use teeny_driver::error::Result;

// #[derive(Default)]
// pub struct CpuDriver {}

// impl CpuDriver {
//     pub const fn new() -> Self {
//         Self {}
//     }
// }

// impl Driver for CpuDriver {
//     fn init(&mut self) -> Result<()> {
//         Ok(())
//     }

//     fn deinit(&mut self) -> Result<()> {
//         // no-op
//         Ok(())
//     }

//     fn id(&self) -> &str {
//         CPU_DRIVER_ID
//     }

//     fn name(&self) -> &str {
//         "Teenygrad CPU Driver v0.1.0"
//     }

//     fn devices(&self) -> Result<Vec<Arc<Mutex<dyn Device>>>> {
//         Ok(vec![])
//     }
// }

// #[ctor]
// fn register_cuda() {
//     DriverManager::register(Arc::new(Mutex::new(CpuDriver::new())));
// }

#[derive(Debug)]
pub struct CpuTensor<T: num::Num> {
    pub data: ndarray::Array<T, IxDyn>,
}

impl<T: num::Num> CpuTensor<T> {
    pub fn new(data: ndarray::Array<T, IxDyn>) -> Self {
        Self { data }
    }
}

// impl Tensor<f32, CpuDevice> for CpuTensor<f32> {
//     type DType = f32;

//     fn add(&self, other: &CpuTensor<f32>) -> CpuTensor<f32> {
//         CpuTensor {
//             data: &self.data + &other.data,
//         }
//     }
// }

// impl Device<f32> for CpuDevice {
//     type Tensor = CpuTensor<f32>;

//     fn from_ndarray(ndarray: ndarray::Array<f32, IxDyn>) -> Self::Tensor {
//         CpuTensor::new(ndarray)
//     }
// }
