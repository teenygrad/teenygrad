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

use crate::{error::Error, torch::Shape};

impl<'a> TryFrom<Shape<'a>> for teeny_core::fxgraph::shape::Shape {
    type Error = Error;

    fn try_from(shape: Shape) -> Result<Self, Self::Error> {
        let shape = shape
            .dims()
            .ok_or_else(|| Error::InvalidBuffer(format!("{shape:?}")))?
            .into_iter()
            .map(|s| s.try_into())
            .collect::<Result<Vec<_>, _>>()?;

        Ok(teeny_core::fxgraph::shape::Shape { shape })
    }
}
