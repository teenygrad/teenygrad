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
