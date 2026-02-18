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

use target_triple::{HOST, TARGET};

use crate::device::{CpuDevice, DeviceProperties};

use crate::error::Result;
use crate::target::CpuTarget;

pub struct CpuDriver;

impl CpuDriver {
    pub fn devices() -> Result<Vec<CpuDevice>> {
        let device = CpuDevice {
            id: "cpu:0".to_string(),
            name: "CPU".to_string(),
            properties: DeviceProperties {
                host: HOST,
                target: CpuTarget::try_from(TARGET)?,
            },
        };

        Ok(vec![device])
    }
}
