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

use crate::{
    tensor::{DenseTensor, Dim2, Tensor},
    types::F32,
};

/// Example demonstrating tensor operations with computation graphs
pub fn tensor_operations_example() {
    // Create some tensors
    let tensor_a = DenseTensor::<Dim2<2, 3>, F32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);

    let tensor_b = DenseTensor::<Dim2<2, 3>, F32>::new(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [2, 3]);

    // Convert to trait objects for dynamic dispatch
    let tensor_a: Box<dyn Tensor<Dim2<2, 3>, Element = F32>> = Box::new(tensor_a);
    let tensor_b: Box<dyn Tensor<Dim2<2, 3>, Element = F32>> = Box::new(tensor_b);

    // Perform operations - these create computation graph nodes
    let result_add = tensor_a + tensor_b; // Creates AddNode
    let result_mul = result_add * 2.0; // Creates ScalarMulNode

    // The operations are lazy - no actual computation happens yet
    // The result is a computation graph that can be evaluated later

    println!("Tensor operations completed!");
    println!("Result shape: {:?}", result_mul.shape());
    println!("Result rank: {}", result_mul.rank());
    println!("Result device: {:?}", result_mul.device());
}

/// Example showing how operations can be chained
pub fn chained_operations_example() {
    // Create tensors
    let tensor_a = DenseTensor::<Dim2<2, 2>, F32>::new(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);

    let tensor_b = DenseTensor::<Dim2<2, 2>, F32>::new(vec![2.0, 1.0, 1.0, 2.0], [2, 2]);

    let tensor_a: Box<dyn Tensor<Dim2<2, 2>, Element = F32>> = Box::new(tensor_a);
    let tensor_b: Box<dyn Tensor<Dim2<2, 2>, Element = F32>> = Box::new(tensor_b);

    // Chain multiple operations
    let result = tensor_a + tensor_b * 3.0 + 1.0;

    // This creates a computation graph like:
    // AddNode {
    //   lhs: tensor_a,
    //   rhs: AddNode {
    //     lhs: MulNode { lhs: tensor_b, rhs: 3.0 },
    //     rhs: 1.0
    //   }
    // }

    println!("Chained operations completed!");
    println!("Final result shape: {:?}", result.shape());
}

/// Example showing in-place operations
pub fn inplace_operations_example() {
    let mut tensor = DenseTensor::<Dim2<2, 2>, F32>::new(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);

    let mut tensor: Box<dyn Tensor<Dim2<2, 2>, Element = F32>> = Box::new(tensor);

    // In-place operations
    tensor += 2.0; // Adds 2.0 to all elements
    tensor *= 3.0; // Multiplies all elements by 3.0

    println!("In-place operations completed!");
    println!("Final tensor shape: {:?}", tensor.shape());
}
