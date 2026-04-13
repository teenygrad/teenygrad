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
    nn::{
        Layer,
        activation::{relu::Relu, softmax::Softmax},
        conv2d::Conv2d,
        flatten::Flatten,
        graph::{DtypeRepr, SymTensor},
        linear::Linear,
        pool::AvgPool2d,
    },
    sequential,
};

fn main() {
    // -----------------------------------------------------------------------
    // Build the LeNet-5 model as a sequential pipeline.
    // Every layer is parameterised by its IO tensor type (SymTensor here) so
    // the same definition compiles for both symbolic tracing and eager
    // execution once the eager backend is implemented.
    // -----------------------------------------------------------------------
    let model = sequential![
        // Block 1 — C1/S2
        Conv2d::<f32, SymTensor, SymTensor, 4>::new(
            1,        // in_channels  (grayscale)
            6,        // out_channels
            (5, 5),   // kernel_size
            (1, 1),   // stride
            (2, 2),   // padding=2 keeps the 28×28 spatial size
            true,     // has_bias
        ),
        Relu::<f32, SymTensor, 4>::new(),
        AvgPool2d::<f32, SymTensor, SymTensor, 4>::new((2, 2), (2, 2)),

        // Block 2 — C3/S4
        Conv2d::<f32, SymTensor, SymTensor, 4>::new(
            6,        // in_channels
            16,       // out_channels
            (5, 5),   // kernel_size
            (1, 1),   // stride
            (0, 0),   // no padding → 14×14 → 10×10
            true,
        ),
        Relu::<f32, SymTensor, 4>::new(),
        AvgPool2d::<f32, SymTensor, SymTensor, 4>::new((2, 2), (2, 2)),

        // Flatten 16×5×5 = 400
        Flatten::<f32, SymTensor, SymTensor>::new(),

        // Classifier — C5/F6/Output
        Linear::<f32, SymTensor, SymTensor, 2>::new(400, 120, true),
        Relu::<f32, SymTensor, 2>::new(),
        Linear::<f32, SymTensor, SymTensor, 2>::new(120, 84, true),
        Relu::<f32, SymTensor, 2>::new(),
        Linear::<f32, SymTensor, SymTensor, 2>::new(84, 10, true),
        Softmax::<f32, SymTensor, 2>::new(1)
    ];

    // -----------------------------------------------------------------------
    // Trace: feed a symbolic input through the model to populate the graph.
    // -----------------------------------------------------------------------
    let (input, graph) = SymTensor::input(DtypeRepr::F32, 4);
    let _output = Layer::call(&model, input);

    // -----------------------------------------------------------------------
    // Inspect the extracted computation graph.
    // -----------------------------------------------------------------------
    let g = graph.borrow();
    let order = g.topological_sort();

    println!("LeNet-5 computation graph ({} nodes)\n", g.nodes.len());
    println!("{:<4}  {:<8}  {:?}", "id", "rank", "op");
    println!("{}", "-".repeat(60));
    for id in &order {
        let node = &g.nodes[*id];
        println!("{:<4}  rank={:<4}  {:?}", id, node.rank, node.op);
    }
}
