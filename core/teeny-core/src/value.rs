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

use std::ops::{Add, Div, Sub};

use crate::types::bf16::bf16;
use crate::types::bool::Bool;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Usize(usize),
    F32(f32),
    Bf16(bf16),
    Bool(Bool),
}

impl From<usize> for Value {
    fn from(value: usize) -> Self {
        Value::Usize(value)
    }
}

impl From<f32> for Value {
    fn from(value: f32) -> Self {
        Value::F32(value)
    }
}

impl From<bf16> for Value {
    fn from(value: bf16) -> Self {
        Value::Bf16(value)
    }
}

impl From<Bool> for Value {
    fn from(value: Bool) -> Self {
        Value::Bool(value)
    }
}

/* -------------------- ARITHMETIC OPERATIONS -------------------- */

impl Add<&Value> for &Value {
    type Output = Value;

    fn add(self, other: &Value) -> Value {
        match (self, other) {
            (Value::Usize(a), Value::Usize(b)) => Value::Usize(a + b),
            (Value::F32(a), Value::F32(b)) => Value::F32(a + b),
            _ => todo!(),
        }
    }
}

impl Sub<&Value> for &Value {
    type Output = Value;

    fn sub(self, other: &Value) -> Value {
        match (self, other) {
            (Value::Usize(a), Value::Usize(b)) => Value::Usize(a - b),
            (Value::F32(a), Value::F32(b)) => Value::F32(a - b),
            (Value::Bf16(_), Value::Bf16(_)) => todo!(),
            _ => todo!(),
        }
    }
}

impl Div<&Value> for &Value {
    type Output = Value;

    fn div(self, other: &Value) -> Value {
        match (self, other) {
            (Value::Usize(a), Value::Usize(b)) => Value::Usize(a / b),
            (Value::F32(a), Value::F32(b)) => Value::F32(a / b),
            (Value::Bf16(_), Value::Bf16(_)) => todo!(),
            _ => todo!(),
        }
    }
}
