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

use crate::{error::Error, torch::DType};

impl TryFrom<DType> for teeny_core::fxgraph::dtype::DType {
    type Error = Error;

    fn try_from(dtype: DType) -> Result<Self, Self::Error> {
        match dtype {
            DType::FLOAT32 => Ok(teeny_core::fxgraph::dtype::DType::F32),
            DType::BFLOAT16 => Ok(teeny_core::fxgraph::dtype::DType::BF16),
            DType::BOOL => Ok(teeny_core::fxgraph::dtype::DType::Bool),
            _ => Err(Error::UnsupportedDtype(dtype)),
        }
    }
}
