# Summary

[Introduction](./introduction.md)

# Orientation

- [What a Kernel Is](./orientation/what-a-kernel-is.md)
- [You Program a Block, Not a Thread](./orientation/block-not-thread.md)
- [From Rust to PTX](./orientation/rust-to-ptx.md)
- [Setting Up](./orientation/setting-up.md)

# Your First Kernel

- [Vector Add, End to End](./first-kernel/vector-add.md)
- [The Kernel Body](./first-kernel/kernel-body.md)
- [Loads, Stores, and Masks](./first-kernel/loads-stores-masks.md)
- [What `#[kernel]` Generates](./first-kernel/kernel-macro.md)
- [Compiling and Reading the Output](./first-kernel/compiling.md)

# Real Patterns

- [Softmax: Your First Reduction](./patterns/softmax.md)
- [Matrix Multiplication](./patterns/matmul.md)
- [Fusing an Epilogue](./patterns/fusion.md)
- [Reductions and Scans](./patterns/reductions.md)
- [Atomics](./patterns/atomics.md)
- [Compile-Time Parameters and Dtype Dispatch](./patterns/specialisation.md)

# Making It Fast

- [Choosing a Block Size](./fast/block-size.md)
- [Memory Coalescing and Tensor Layout](./fast/layout.md)
- [Measuring](./fast/measuring.md)
- [Numerics](./fast/numerics.md)

# Kernels in a Real Model

- [Your Kernel as a Graph Op](./in-a-model/graph-op.md)
- [Wiring the Runtime](./in-a-model/runtime-op.md)
- [Training: The Backward Kernel](./in-a-model/backward.md)
- [Building for Another Target](./in-a-model/cross-building.md)
- [What Is Portable](./in-a-model/portability.md)

# Reference

- [Python Triton to Rust](./reference/translation-table.md)
- [Common Compile Errors](./reference/compile-errors.md)
- [Glossary](./reference/glossary.md)
- [Appendix: Porting a Python Triton Kernel](./reference/porting.md)
