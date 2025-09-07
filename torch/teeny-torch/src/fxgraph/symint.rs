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

use crate::{error::Error, torch::SymInt};

impl<'a> TryFrom<SymInt<'a>> for teeny_core::fxgraph::shape::SymInt {
    type Error = Error;

    fn try_from(symint: SymInt) -> Result<Self, Self::Error> {
        let value = symint
            .value()
            .ok_or_else(|| Error::InvalidBuffer(format!("{symint:?}")))?;

        let sym_int = value.parse::<i64>().ok();
        if let Some(sym_int) = sym_int {
            Ok(teeny_core::fxgraph::shape::SymInt::Int(sym_int))
        } else {
            Ok(teeny_core::fxgraph::shape::SymInt::Sym(value.to_string()))
        }
    }
}
