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

use egg::Id;
use ordered_float::OrderedFloat;

use crate::{error::Error, fxgraph::dtype::DType};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Value {
    None,
    Ellipsis,
    Int(i64),
    DType(DType),
    Float64(OrderedFloat<f64>),
    String(String),
    Device(String),
    Bool(bool),
    Node(Id),
    Tuple(Vec<Box<Value>>),
    List(Vec<Box<Value>>),
    Slice(Box<Value>, Box<Value>, Box<Value>),
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::None => write!(f, "None"),
            Value::Ellipsis => write!(f, "..."),
            Value::Int(i) => write!(f, "{}", i),
            Value::DType(dtype) => write!(f, "{}", dtype),
            Value::Float64(fx) => write!(f, "{}", fx),
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
        }
    }
}

impl FromStr for Value {
    type Err = Error;

    fn from_str(_s: &str) -> Result<Self, Self::Err> {
        todo!()
    }
}
