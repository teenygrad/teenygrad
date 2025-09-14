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
use ordered_float::OrderedFloat;

use crate::fxgraph::{analysis::GraphAnalysis, lang::FxGraphLang, types::ty_tensor::TyTensor};

use crate::{
    error::Error,
    fxgraph::{
        dtype::DType,
        shape::SymInt,
        tensor::Tensor,
        types::{Type, TypeInfo},
    },
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Value {
    None,
    Ellipsis,
    SymInt(SymInt),
    Int64(i64),
    DType(DType),
    Float32(OrderedFloat<f32>),
    String(String),
    Device(String),
    Bool(bool),
    Node(Id),
    Tensor(Tensor),
    Tuple(Vec<Box<Value>>),
    List(Vec<Box<Value>>),
    Slice(Box<Value>, Box<Value>, Box<Value>),
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::None => write!(f, "None"),
            Value::Ellipsis => write!(f, "..."),
            Value::Int64(i) => write!(f, "{}", i),
            Value::DType(dtype) => write!(f, "{}", dtype),
            Value::Float32(fx) => write!(f, "{}", fx),
            Value::String(s) => write!(f, "{}", s),
            Value::Device(d) => write!(f, "{}", d),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Node(n) => write!(f, "{}", n),
            Value::Tuple(t) => {
                write!(f, "(")?;
                for (i, v) in t.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, ")")?;
                Ok(())
            }
            Value::List(l) => {
                write!(f, "[")?;
                for (i, v) in l.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")?;
                Ok(())
            }
            Value::Slice(s1, s2, s3) => write!(f, "slice({},{},{})", s1, s2, s3),
            Value::SymInt(s) => write!(f, "{:?}", s),
            Value::Tensor(t) => write!(f, "{:?}", t),
        }
    }
}

impl FromStr for Value {
    type Err = Error;

    fn from_str(_s: &str) -> Result<Self, Self::Err> {
        todo!()
    }
}

impl Value {
    pub fn depends_on(&self) -> Vec<Id> {
        match self {
            Value::Node(id) => vec![*id],
            Value::List(ids) => ids.iter().flat_map(|id| id.depends_on()).collect(),
            Value::Tuple(values) => values.iter().flat_map(|value| value.depends_on()).collect(),
            Value::Slice(value1, value2, value3) => {
                let mut depends_on = value1.depends_on();
                depends_on.extend(value2.depends_on());
                depends_on.extend(value3.depends_on());
                depends_on
            }
            Value::None => vec![],
            Value::Ellipsis => vec![],
            Value::SymInt(_) => vec![],
            Value::Int64(_) => vec![],
            Value::DType(_) => vec![],
            Value::Float32(_) => vec![],
            Value::String(_) => vec![],
            Value::Device(_) => vec![],
            Value::Bool(_) => vec![],
            Value::Tensor(_) => vec![],
        }
    }
}

impl TypeInfo for Value {
    fn ty(&self, egraph: &mut EGraph<FxGraphLang, GraphAnalysis>) -> Result<Type, Error> {
        let ty = match self {
            Value::SymInt(_) => Type::SymInt,
            Value::Tensor(t) => Type::Tensor(TyTensor::new(t)),
            Value::Node(id) => id.ty(egraph)?,
            Value::List(values) => Type::List(
                values
                    .iter()
                    .map(|value| value.ty(egraph))
                    .collect::<Result<Vec<Type>, Error>>()?,
            ),
            Value::Tuple(values) => Type::Tuple(
                values
                    .iter()
                    .map(|value| value.ty(egraph))
                    .collect::<Result<Vec<Type>, Error>>()?,
            ),
            _ => todo!("unsupported value: {self:?}"),
        };

        Ok(ty)
    }
}
