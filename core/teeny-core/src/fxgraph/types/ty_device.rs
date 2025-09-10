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

use z3::{DatatypeAccessor, DatatypeBuilder, Sort, ast::Dynamic};

use crate::fxgraph::{device::Device, types::TypeTheory};

pub fn device_builder() -> DatatypeBuilder {
    DatatypeBuilder::new("Device")
        .variant(
            "CPU",
            vec![("value", DatatypeAccessor::Sort(Sort::string()))],
        )
        .variant(
            "GPU",
            vec![("value", DatatypeAccessor::Sort(Sort::string()))],
        )
}

pub fn create_device_ty(th: &mut TypeTheory, device: &Device) -> Dynamic {
    let (constructor, value) = match device {
        Device::Cpu(value) => (&th.device_sort.variants[0].constructor, value),
        Device::Cuda(value) => (&th.device_sort.variants[1].constructor, value),
    };

    let value = z3::ast::String::new_const(value.clone());
    constructor.apply(&[&value])
}
