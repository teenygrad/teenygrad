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

use std::{
    fmt::{Display, Formatter},
    str::FromStr,
};

use egg::{EGraph, Id};

use crate::{
    error::Error,
    fxgraph::{
        analysis::GraphAnalysis,
        lang::FxGraphLang,
        types::{Type, TypeInfo},
    },
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Add {
    pub lhs: Id,
    pub rhs: Id,
}

impl Display for Add {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl FromStr for Add {
    type Err = Error;

    fn from_str(_s: &str) -> Result<Self, Self::Err> {
        todo!()
    }
}

impl TypeInfo for Add {
    fn ty(&self, _egraph: &mut EGraph<FxGraphLang, GraphAnalysis>) -> Result<Type, Error> {
        todo!()
    }
}
