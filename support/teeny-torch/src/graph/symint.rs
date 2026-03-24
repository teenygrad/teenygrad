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

use crate::{error::Error, torch::SymInt};

impl<'a> TryFrom<SymInt<'a>> for teeny_fxgraph::fxgraph::shape::SymInt {
    type Error = Error;

    fn try_from(symint: SymInt) -> Result<Self, Self::Error> {
        let value = symint
            .value()
            .ok_or_else(|| Error::InvalidBuffer(format!("{symint:?}")))?;

        let sym_int = value.parse::<i64>().ok();
        if let Some(sym_int) = sym_int {
            Ok(teeny_fxgraph::fxgraph::shape::SymInt::Int(sym_int))
        } else {
            Ok(teeny_fxgraph::fxgraph::shape::SymInt::Sym(
                value.to_string(),
            ))
        }
    }
}
