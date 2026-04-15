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

//! LeNet-5 — Yann LeCun's convolutional network for MNIST (1998).
//!
//! Original paper: "Gradient-Based Learning Applied to Document Recognition"
//! LeCun, Bottou, Bengio, Haffner (1998).
//!
//! Architecture (adapted to use ReLU in place of the original tanh/sigmoid):
//!
//! ```text
//! Input         [N,  1, 28, 28]
//! Conv2d(1→6,   5×5, pad=2)  →  [N,  6, 28, 28]   (same-padding keeps spatial dims)
//! ReLU
//! AvgPool2d(2×2, stride=2)   →  [N,  6, 14, 14]
//! Conv2d(6→16,  5×5, pad=0)  →  [N, 16, 10, 10]
//! ReLU
//! AvgPool2d(2×2, stride=2)   →  [N, 16,  5,  5]
//! Flatten                    →  [N, 400]
//! Linear(400→120)
//! ReLU
//! Linear(120→84)
//! ReLU
//! Linear(84→10)
//! Softmax(dim=1)             →  [N, 10]  class probabilities
//! ```
//!
//! This example traces the model symbolically using `SymTensor`, extracts the
//! computation graph, and prints every node in topological order.

use teeny_core::{
    dtype::Float,
    graph::SymTensor,
    nn::{
        Layer,
        activation::{relu::Relu, softmax::Softmax},
        conv2d::Conv2d,
        flatten::Flatten,
        linear::Linear,
        pool::AvgPool2d,
    },
    sequential,
};

pub fn mnist<D: Float>() -> impl Fn(SymTensor) -> SymTensor {
    // -----------------------------------------------------------------------
    // Build the LeNet-5 model as a sequential pipeline.
    // Every layer is parameterised by its IO tensor type (SymTensor here) so
    // the same definition compiles for both symbolic tracing and eager
    // execution once the eager backend is implemented.
    // -----------------------------------------------------------------------
    sequential![
        // Block 1 — C1/S2
        Conv2d::<D, _, _, 4>::new(
            1,      // in_channels  (grayscale)
            6,      // out_channels
            (5, 5), // kernel_size
            (1, 1), // stride
            (2, 2), // padding=2 keeps the 28×28 spatial size
            true,   // has_bias
        ),
        Relu::<D, _, 4>::new(),
        AvgPool2d::<D, _, _, 4>::new((2, 2), (2, 2)),
        // Block 2 — C3/S4
        Conv2d::<D, _, _, 4>::new(
            6,      // in_channels
            16,     // out_channels
            (5, 5), // kernel_size
            (1, 1), // stride
            (0, 0), // no padding → 14×14 → 10×10
            true,
        ),
        Relu::<D, _, 4>::new(),
        AvgPool2d::<D, _, _, 4>::new((2, 2), (2, 2)),
        // Flatten 16×5×5 = 400
        Flatten::<D, _, _>::new(),
        // Classifier — C5/F6/Output
        Linear::<D, _, _, 2>::new(400, 120, true),
        Relu::<D, _, 2>::new(),
        Linear::<D, _, _, 2>::new(120, 84, true),
        Relu::<D, _, 2>::new(),
        Linear::<D, _, _, 2>::new(84, 10, true),
        Softmax::<D, _, 2>::new(1)
    ]
}
