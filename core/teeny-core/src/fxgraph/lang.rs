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

use crate::fxgraph::dtype::DtypeValue;
use crate::fxgraph::literal::ConstantValue;
use crate::fxgraph::shape::ShapeValue;

// Define the core language for your compiler IR
define_language! {
    pub enum FxGraphLang {
        // Constants and literals
        Constant(ConstantValue),
        Shape(ShapeValue),
        Dtype(DtypeValue),
        Symbol(egg::Symbol),

        // Placeholders
        Placeholder(String),

        // Tensor
        "tensor" = Tensor([Id; 2]), // [shape_id, dtype_id]

        // Pytorch ops
        "log_api_usage_once" = LogApiUsageOnce([Id; 0]),
        "sdpa" = ScaledDotProductAttention([Id; 3]),
        "add_batch_dim" = AddBatchDim(Id),
        "remove_batch_dim" = RemoveBatchDim(Id),
        "vmap_decrement_nesting" = VmapDecrementNesting([Id; 0]),
        "vmap_increment_nesting" = VmapIncrementNesting([Id; 0]),
        "enter_autocast" = EnterAutocast([Id; 0]),
        "exit_autocast" = ExitAutocast([Id; 0]),
        "lazy_load_decompositions" = LazyLoadDecompositions([Id; 0]),
        "+" = Add([Id; 2]),
        "/" = Div([Id; 2]),
        "-" = Sub([Id; 2]),
        "iadd" = IAdd([Id; 2]),
        "linear" = Linear([Id; 2]),
        "@" = MatMul([Id; 2]),
        "*" =  Mul([Id; 2]),
        "neg" = Neg(Id),
        "arange" = Arange([Id; 3]),
        "cat" = Cat([Id; 2]),
        "rsqrt" = Rsqrt(Id),
        "embedding" = Embedding([Id; 5]),
        "silu" = Silu(Id),
        "symsum" = SymSum(Id),
        "aten.index" = AtenIndex([Id; 2]),
        "contiguous" = Contiguous(Id),
        "cos" = Cos(Id),
        "expand" = Expand([Id; 2]),
        "float" = Float(Id),
        "item" = Item(Id),
        "le" = Le([Id; 2]),
        "mean" = Mean(Id),
        "new_ones" = NewOnes(Id),
        "numel" = Numel(Id),
        "pow" = Pow([Id; 2]),
        "reshape" = Reshape(Id),
        "sin" = Sin(Id),
        "to" = To(Id),
        "transpose" = Transpose(Id),
        "unsqueeze" = Unsqueeze(Id),
        "view" = View(Id),
    }
}

pub fn const_bool(value: bool) -> FxGraphLang {
    FxGraphLang::Constant(ConstantValue::Bool(value))
}

pub fn const_i64(value: i64) -> FxGraphLang {
    FxGraphLang::Constant(ConstantValue::Int(value))
}

pub fn const_string(value: &str) -> FxGraphLang {
    FxGraphLang::Constant(ConstantValue::String(value.to_string()))
}
