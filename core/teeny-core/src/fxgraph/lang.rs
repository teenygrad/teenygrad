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

        // Tensors and basic operations
        "tensor" = Tensor([Id; 2]), // [shape_id, dtype_id]

        // Placeholders
        "placeholder" = Placeholder([Id; 1]), // [shape_id, dtype_id]

        // Operations
        "linear" = Linear([Id; 3]), // [input_id, weight_id, bias_id]
        "relu" = Relu(Id),


        // Tensor operations
        "+" = Add([Id; 2]), // [a_id, b_id]
        "*" = Mul([Id; 2]), // [a_id, b_id]
        "-" = Sub([Id; 2]), // [a_id, b_id]
        "/" = Div([Id; 2]), // [a_id, b_id]
        "@" = MatMul([Id; 2]), // [a_id, b_id]

        // Tensor operations
        "contiguous" = Contiguous(Id),
        "transpose" = Transpose(Id),
    }
}
