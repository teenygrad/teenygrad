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
