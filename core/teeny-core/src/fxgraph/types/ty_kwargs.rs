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

use egg::EGraph;

use crate::{
    error::Error,
    fxgraph::{
        analysis::GraphAnalysis,
        keyvalue::{KeyValue, KeyValueList},
        lang::FxGraphLang,
        types::{Type, TypeInfo},
    },
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TyKwArgs {
    pub args: Vec<(String, Type)>,
}

impl TyKwArgs {
    pub fn new(
        egraph: &mut EGraph<FxGraphLang, GraphAnalysis>,
        args: &KeyValueList,
    ) -> Result<Self, Error> {
        Ok(TyKwArgs {
            args: args
                .0
                .iter()
                .map(|KeyValue::Kv(key, value)| value.ty(egraph).map(|ty| (key.clone(), ty)))
                .collect::<Result<Vec<(String, Type)>, Error>>()?,
        })
    }
}
