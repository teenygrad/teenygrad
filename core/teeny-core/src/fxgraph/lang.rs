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

use crate::fxgraph::keyvalue::KeyValueList;
use crate::fxgraph::literal::ConstantValue;
use crate::fxgraph::placeholder::Placeholder;
use crate::fxgraph::value::Value;

pub type Node = FxGraphLang;

// Define the core language for your compiler IR
define_language! {
    pub enum FxGraphLang {
        Constant(ConstantValue),
        Placeholder(Placeholder),
        Value(Value),
        KwArgs(KeyValueList),

        // ops
        "output" = Output([Id; 2]),  // arg0: list of args, arg1: kwargs
        "add" = Add([Id; 2]),
        "list" = List(Vec<Id>),
        "args" = Args(Vec<Id>),
        "item" = ItemMethod(Vec<Id>), // item method
        "getitem" = GetItem([Id; 2]),
        "embedding" = Embedding([Id; 7]),
        "arange" = Arange(Vec<Id>),
        "iadd" = IAdd([Id; 2]),
        "lazy_load_decompositions" = LazyLoadDecompositions([Id; 0]),
        "contiguous" = Contiguous(Id),
        "enter_autocast" = EnterAutocast(Vec<Id>),
        "exit_autocast" = ExitAutocast([Id; 1]),
        "silu" = Silu(Vec<Id>),
        "rsqrt" = Rsqrt([Id; 1]),
        "linear" = Linear(Vec<Id>),
        "sym_sum" = SymSum(Id),
        "vmap_increment_nesting" = VmapIncrementNesting([Id; 2]),
        "vmap_decrement_nesting" = VmapDecrementNesting([Id; 0]),
        "add_batch_dim" = AddBatchDim([Id; 3]),
        "remove_batch_dim" = RemoveBatchDim([Id; 4]),
        "aten.index" = AtenIndex([Id; 2]),
        "cat" = Cat(Vec<Id>),
        "sdpa" = ScaledDotProductAttention(Vec<Id>),
        "-" = Sub([Id; 2]),
        "*" =  Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "@" = MatMul([Id; 2]),
        "neg" = Neg(Id),
        "to" = To(Vec<Id>),
        "new_ones" = NewOnes(Vec<Id>),
        "le" = Le(Vec<Id>),
        "and" = And(Vec<Id>),
        "float" = Float([Id; 1]),
        "expand" = Expand(Vec<Id>),
        "transpose" = Transpose(Vec<Id>),
        "cos" = Cos([Id; 1]),
        "sin" = Sin([Id; 1]),
        "pow" = Pow([Id; 2]),
        "mean" = Mean(Vec<Id>),
        "view" = View([Id; 2]),
        "unsqueeze" = Unsqueeze([Id; 2]),
        "reshape" = Reshape(Vec<Id>),
        "numel" = Numel(Id),
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
