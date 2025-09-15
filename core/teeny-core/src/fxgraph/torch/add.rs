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

use egg::{EGraph, Id};

use crate::{
    error::Error,
    fxgraph::{
        analysis::GraphAnalysis,
        lang::FxGraphLang,
        types::{Type, TypeInfo},
    },
};

pub fn add_ty(
    egraph: &mut EGraph<FxGraphLang, GraphAnalysis>,
    args: &[Id; 2],
) -> Result<Type, Error> {
    let lhs = args[0].ty(egraph)?;
    let rhs = args[1].ty(egraph)?;

    match (&lhs, &rhs) {
        (Type::Tensor(lhs), Type::Tensor(rhs)) => {
            if lhs.device != rhs.device {
                return Err(Error::InvalidDevice(format!(
                    "Mismatched devices: {:?} and {:?}",
                    lhs.device, rhs.device
                )));
            }

            let result = lhs.broadcast(rhs)?;
            Ok(Type::Tensor(result))
        }
        _ => todo!("unsupported types: {lhs:?} and {rhs:?}"),
    }
}
