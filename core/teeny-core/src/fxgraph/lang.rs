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

use egg::{Id, define_language};
use ordered_float::OrderedFloat;

use crate::fxgraph::keyvalue::KeyValue;
use crate::fxgraph::literal::ConstantValue;
use crate::fxgraph::placeholder::Placeholder;

// Define the core language for your compiler IR
define_language! {
    pub enum FxGraphLang {
        // Constants and literals
        Constant(ConstantValue),

        // Placeholders
        Placeholder(Placeholder),

        // Pytorch ops
        "tuple" = Tuple(Vec<Id>),
        "embedding" = Embedding([Id; 5]),
        "arange" = Arange([Id; 3]),
        "iadd" = IAdd([Id; 2]),
        "lazy_load_decompositions" = LazyLoadDecompositions([Id; 0]),
        "getitem" = GetItem([Id; 2]),
        "contiguous" = Contiguous(Id),
        "enter_autocast" = EnterAutocast([Id; 2]),
        "exit_autocast" = ExitAutocast([Id; 1]),
        "silu" = Silu([Id; 2]),
        "rsqrt" = Rsqrt([Id; 1]),
        "linear" = Linear([Id; 2]),
        "sym_sum" = SymSum(Vec<Id>),
        "vmap_increment_nesting" = VmapIncrementNesting([Id; 2]),
        "vmap_decrement_nesting" = VmapDecrementNesting([Id; 0]),
        "add_batch_dim" = AddBatchDim([Id; 3]),
        "remove_batch_dim" = RemoveBatchDim([Id; 4]),
        "aten.index" = AtenIndex([Id; 2]),
        "cat" = Cat([Id; 2]),
        "output" = Output(Vec<Id>),
        "sdpa" = ScaledDotProductAttention([Id; 7]),
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" =  Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "@" = MatMul([Id; 2]),
        "neg" = Neg(Id),

    }
}

pub fn const_bool(value: &str) -> FxGraphLang {
    FxGraphLang::Constant(ConstantValue::Bool(value.to_lowercase().parse().unwrap()))
}

pub fn const_i64(value: i64) -> FxGraphLang {
    FxGraphLang::Constant(ConstantValue::Int(value))
}

pub fn const_f32(value: f32) -> FxGraphLang {
    FxGraphLang::Constant(ConstantValue::Float32(OrderedFloat::from(value)))
}

pub fn const_string(value: &str) -> FxGraphLang {
    FxGraphLang::Constant(ConstantValue::String(value.to_string()))
}

pub fn const_kv(key: &str, value: Id) -> KeyValue {
    KeyValue::Kv(key.to_string(), value)
}
