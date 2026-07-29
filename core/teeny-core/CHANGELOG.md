# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0](https://github.com/teenygrad/teenygrad/releases/tag/teeny-core-v0.1.0) - 2026-07-29

### Added

- add ONNX FlexAttention/LinearAttention/CausalConvWithState support
- make flash_attn2 generic over Float
- add dtype-aware kernel dispatch and generic float activations
- add ahead-of-time kernel compilation support
- *(teeny-onnx)* add full ONNX operator coverage to graph IR
- name PTX entry points after the kernel function
- fused Conv2d+BN+SiLU kernel and Graph::optimise() pass
- RuntimeOp impls for GELU/Sigmoid/LayerNorm and concrete shape propagation
- trim Capability enum to sm_75+, add sm_87 and sm_120
- add graph node naming infrastructure for safetensors weight loading
- add CustomOp trait, lowering middleware chain, and migrate detect_decode to vision-rs
- add CustomOp trait with shape inference and lowering middleware chain
- add channel_bias_add and detect_decode kernels; fix grouped Conv2d lowering
- add MaxPool2d padding support and real RuntimeOp
- add ChannelCat/ChannelChunk RuntimeOp impls with multi-launch executor
- add UpsampleNearest2d kernel with forward/backward, graph node, and tests
- add groups to Conv2d op and Op::Attention for C2PSA graph
- add supporting infrastructure for Flash Attention 2 backward
- add ChannelChunk, ChannelCat, and Add graph ops with CUDA kernels
- add norm kernels (GroupNorm/InstanceNorm/LayerNorm/RMSNorm/BatchNorm) and fix variance masking bug
- graph compiler with CUDA runtime and MNIST training ([#8](https://github.com/teenygrad/teenygrad/pull/8))
- wip graph compiler
- *(teeny-triton)* add scalar comparison ops and refresh kitchen sink snapshots
- split kernel source fields and return sha256 id as hex string
- track output shape on SymTensor and GraphNode
- Modified license to Apache 2.0 to make it fully opensource

### Fixed

- resolve all clippy warnings, tighten CI to -D warnings + fmt --check
- use NCHW-native BN kernel in training mode; add param_names to normalize op
- resolve conv2d/avgpool2d test crashes and update snapshots
- fixed adam optimizer tests
- fixed incorrect copyright notice

### Other

- complete public-API doc coverage for teeny-core
- structural doc coverage for teeny-core (modules/structs/traits/enums)
- apply cargo fmt --all workspace-wide
- add crate-level //! docs to every publishable crate
- prepare workspace for crates.io publishing and self-hosted docs
- move Float::to_le_bytes onto host-only FloatBytes
- centralise workspace package metadata
- move PSA attention from Op::Attention lowering to CustomOp in vision-rs
- embed CustomOp lowering into trait, handle Op::Custom in TritonLowering
- refactor teeny-core crates
- minor refactor and wip model
- remove unused kernels and clean up module structure
- add LeNet-5 MNIST demo with AvgPool2d and Flatten layers
- add Conv2d layer with graph recording support
- add topological sort to Graph using Kahn's algorithm
- add SymTensor graph extraction and EagerTensor marker trait
- add rank/shape generics to dtype traits and nn layer foundations
- added cuda program launcher
- modified compiler api and integration test
- wip integrate compilation support
- wip integrate compilation support
- updated compiler to accept a kernel
- updated macro to generate a kernel struct and implementation
- wip new no_std core
- extracted fxgraph into it's own crate
- rust compiler as external dependency
- updated project dependencies
- wip initial triton dsl
- added device to tensor type
- wip triton dsl
- added support for basic type inference
- wip tensor type inference
- wip type inference for vector addition
- wip type inference of simple network
- wip type inference
- wip type inference
- wip fxgraph type inference
- wip type inference
- wip typecheck fxgraph
- added types and basic axioms
- modified to use unknown analysis
- wip type system
- wip simple vector addition type inference
- wip fxgraph type inference
- wip type theory
- wip z3 based type system
- wip egraph analysis and type checking.
- wip simple vector addition fxgraph
- basic qwen3 graph conversion
- wip fxgraph conversion
- wip convert fxgraph
- wip convert fxgraph
- wip conversion to fxgraph
- wip convert torch graph to fxgraph
- wip torch graph to fxgraph
- wip fxgraph conversion
- wip mapping flat buffers to fxgraph
- wip enhaced fxgraph schema
- initial fxgraph for qwen3
- wip convert torch fxgraph to egraph
- wip convert fxgraph to egraph
- wip convert fxgraph to egraph
- wip convert pytorch graph to egraph
- wip convert fxgraph to egraph
- moved fxgraph to core
- wip qwen3 graph
- wip qwen3 graph
- wip qwen3 graph
- wip qwen3 graph
- wip qwen3 graph
- added sprint review and the next sprint plan
- wip qwen3 graph
- qwen3 graph
- wip qwen3 graph
- wip qwen3 graph
- wip qwen3 graph
- wip mixed precision graph api
- wip mixed precision neural network graph
- wip qwen3 graph
- wip qwen3 embedding module
- wip qwen3 pre-trained model
- modified to use anyhow results for better stack traces
- wip load qwen3 from pre trained
- wip load hugging face model from safetensors
- added support for reading safetensors from the file system
- wip read safetensors data
- wip load hugging face qwen3 bfp16 model
- wip qwen3 cpu
- wip qwen3
- wip qwen 3 using ndarrays
- wip get qwen3 working using ndarrays
- wip compiler api
- wip simple classifier
- wip compile modules
- wip adam optim
- wip adam optim
- wip adam optim
- wip optimizer and loss function
- wip adam impl
- wip simple classifier model definition
- wip cpu based graph implermentation
- wip compile graph to ndarray implementation
- wip compile graph to appropriate target
- wip CISC graph api
- wip basic vector add
- wip graph rep of tensor
- wip ndarray based tensor
- wip ndarray based tensors
- wip tensor ops
- wip tensor ops
- basic wiring up of devices
- wip static dispatch
- wip static dispatch
- wip jit macros
- wip model jit
- wip compile simple module
- added teeny runtime crate
- wip device indepedent tensors
- wip new tensor abstraction for device indepedent tensors
- wip teeny jit
- adding basic wiring up of drivers based on features
- wip adam optimizer
- added tracing based logging to cli projects
- wip backprop for simple classifier
- wip backprop
- wip backprop
- wip nackprop
- wip simple classifier
- wip bce loss calc
- wip simple classifier
- wip backprop
- wip backprop
- wip backprop
- wip backprop
- wip backprop
- wip backprop
- wip backprop
- added tensor std ops for sugar in creating expressions
- wip autograd
- wip autograd
- re-factored to extract ops into separate modules
- wip auto differentiation
- wip simple classifier
- wip simple classifier
- commented out existing code for new tensor implementation
- wip tensor ops
- wip tensor operations for simpl;e classifier
- wip simple classifier
- wip simple classifier
- wip qwen3 model
- wip qwen3 miodel
- copyright change and wip implmenet tiktoken tokenizer

### Removed

- removed z3 based type inference
- removed unused imports
- removed unused module and re-organised folder structure
