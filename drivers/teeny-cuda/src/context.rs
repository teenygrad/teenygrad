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

use std::marker::PhantomData;

use teeny_core::context::{Context, DeviceInfo};

use crate::{
    cuda,
    device::{CudaDevice, CudaDeviceInfo},
    errors::{Error, Result},
};

pub struct Cuda<'a> {
    _unused: PhantomData<&'a ()>,
}

impl<'a> Cuda<'a> {
    pub fn try_new() -> Result<Self> {
        Self::is_available().and_then(|is_available| {
            if !is_available {
                return Err(Error::CudaNotAvailable.into());
            }

            let flags = 0;
            let status = unsafe { cuda::cuInit(flags) };
            if status != cuda::cudaError_enum_CUDA_SUCCESS {
                return Err(Error::from_cuda_error(status).into());
            }

            Ok(Self {
                _unused: PhantomData,
            })
        })
    }

    pub fn is_available() -> Result<bool> {
        let mut device_count = 0;
        let err = unsafe { cuda::cudaGetDeviceCount(&mut device_count) };
        if err != cuda::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::from_cuda_error(err).into());
        }

        Ok(device_count > 0)
    }
}

impl<'a> Context<'a> for Cuda<'a> {
    type Device = CudaDevice<'a>;
    type DeviceInfo = CudaDeviceInfo;

    fn list_devices(&self) -> Result<Vec<Self::DeviceInfo>> {
        let mut devices = Vec::new();
        let mut device_count = 0;
        let err = unsafe { cuda::cudaGetDeviceCount(&mut device_count) };
        if err != cuda::cudaError_enum_CUDA_SUCCESS {
            return Err(Error::from_cuda_error(err).into());
        }

        for id in 0..device_count {
            let mut props = cuda::cudaDeviceProp::default();
            let err = unsafe { cuda::cudaGetDeviceProperties(&mut props, id) };
            if err != cuda::cudaError_enum_CUDA_SUCCESS {
                return Err(Error::from_cuda_error(err).into());
            }

            let device_info = CudaDeviceInfo::new(id, props);
            devices.push(device_info);
        }

        Ok(devices)
    }

    fn device(&self, id: &<Self::DeviceInfo as DeviceInfo>::Id) -> Result<Self::Device> {
        CudaDevice::try_new(*id)
    }
}
