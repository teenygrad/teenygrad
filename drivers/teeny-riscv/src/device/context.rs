/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
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

use teeny_core::device::context::{Context, DeviceInfo};
use teeny_core::device::hardware::HardwareProfile;

use crate::device::RiscvDevice;
use crate::errors::Result;

/// A RISC-V "device"'s identifying metadata. Synthetic: unlike CUDA there is no hardware to
/// query -- there is exactly one "device", the local machine (native RISC-V) or the
/// `qemu-riscv64` process this code is running under/targeting.
#[derive(Debug, Clone)]
pub struct RiscvDeviceInfo {
    id: i32,
    name: String,
}

impl Default for RiscvDeviceInfo {
    fn default() -> Self {
        Self {
            id: 0,
            name: "riscv64 host".to_string(),
        }
    }
}

impl DeviceInfo for RiscvDeviceInfo {
    type Id = i32;

    fn id(&self) -> Self::Id {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    /// Reports only what this synthetic device actually knows, which is little beyond its
    /// name: there is no hardware to query here (see [`RiscvDeviceInfo`]), and this crate's
    /// policy is real hardware data or nothing, never a guess -- the same rule `teeny-cuda`
    /// follows when it leaves bandwidth/latency `None`.
    ///
    /// `execution` is `None` rather than a fabricated warp width. RVV has no warp/wavefront
    /// concept at all: vector length is runtime-queried via `vsetvli` and the effective
    /// element count varies with `SEW`/`LMUL`, so `simt_width` cannot be a single fixed
    /// number here -- exactly the case `ExecutionProfile`'s own doc comment calls out as
    /// needing to be computed per-dtype at codegen time instead.
    ///
    /// `memory_levels` is empty for the same reason: nothing about this target's hierarchy
    /// is queryable, so a cost-model search finds no levels rather than being handed invented
    /// capacities. `compute_units` is `1` -- the single synthetic device described above, not
    /// a claim about any real core count.
    fn hardware_profile(&self) -> HardwareProfile {
        HardwareProfile {
            name: self.name.clone(),
            compute_units: 1,
            memory_levels: Vec::new(),
            execution: None,
        }
    }
}

/// The RISC-V [`Context`]: entry point for listing/opening devices.
pub struct Riscv<'a> {
    _unused: PhantomData<&'a ()>,
}

impl<'a> Riscv<'a> {
    /// Creates the RISC-V context. Infallible -- unlike CUDA there is no driver to initialize.
    pub fn try_new() -> Result<Self> {
        Ok(Self {
            _unused: PhantomData,
        })
    }
}

impl<'a> Context<'a> for Riscv<'a> {
    type Device = RiscvDevice<'a>;
    type DeviceInfo = RiscvDeviceInfo;

    /// Always returns the single synthetic device -- see [`RiscvDeviceInfo`]'s doc comment.
    fn list_devices(&self) -> Result<Vec<Self::DeviceInfo>> {
        Ok(vec![RiscvDeviceInfo::default()])
    }

    fn device(&self, id: &<Self::DeviceInfo as DeviceInfo>::Id) -> Result<Self::Device> {
        Ok(RiscvDevice::new(RiscvDeviceInfo {
            id: *id,
            ..RiscvDeviceInfo::default()
        }))
    }
}
