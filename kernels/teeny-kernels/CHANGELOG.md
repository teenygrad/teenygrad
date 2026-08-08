# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.4](https://github.com/teenygrad/teenygrad/compare/teeny-kernels-v0.1.3...teeny-kernels-v0.1.4) - 2026-08-08

### Other

- release v0.1.3 ([#17](https://github.com/teenygrad/teenygrad/pull/17))

## [0.1.3](https://github.com/teenygrad/teenygrad/compare/teeny-kernels-v0.1.2...teeny-kernels-v0.1.3) - 2026-08-08

### Added

- *(graph)* add Op::Fused and an opt-in elementwise-chain fusion pass ([#19](https://github.com/teenygrad/teenygrad/pull/19))

### Fixed

- *(kernels)* unblock kernel compilation under rustc 1.97.1/LLVM 22/Triton 3.7.1
- *(ci)* install the CUDA toolkit so Build/Clippy/Doc cover teeny-cuda and teeny-kernels ([#18](https://github.com/teenygrad/teenygrad/pull/18))

### Other

- *(kernels)* bless MLIR snapshots for loop-detector simplification
- *(kernels)* pick GEMM tile size (BLOCK_M/N/K) per shape ([#20](https://github.com/teenygrad/teenygrad/pull/20))
- *(kernels)* fuse Conv2d+bias into one kernel for inference ([#21](https://github.com/teenygrad/teenygrad/pull/21))
- *(kernels)* make T::dot's InputPrecision required, not optional
- *(kernels)* remove unused Capability imports across test files
- *(kernels)* rewrite gemm.rs as a tiled GEMM using T::dot for tensor cores
- release v0.1.2 ([#16](https://github.com/teenygrad/teenygrad/pull/16))

## [0.1.2](https://github.com/teenygrad/teenygrad/compare/teeny-kernels-v0.1.1...teeny-kernels-v0.1.2) - 2026-08-06

### Other

- merge release/0.1.2 into main

## [0.1.1](https://github.com/teenygrad/teenygrad/compare/teeny-kernels-v0.1.0...teeny-kernels-v0.1.1) - 2026-08-03

### Other

- merge release/0.1.1 into main
- release v0.1.0 ([#10](https://github.com/teenygrad/teenygrad/pull/10))

## [0.1.0](https://github.com/teenygrad/teenygrad/releases/tag/teeny-kernels-v0.1.0) - 2026-08-02

### Added

- *(kernels)* add criterion benchmarks for conv2d_bn_silu variants
- *(compiler)* auto-detect teenyc via rustup when TEENYC_PATH is unset
- add ONNX FlexAttention/LinearAttention/CausalConvWithState support
- make flash_attn2 generic over Float
- add dtype-aware kernel dispatch and generic float activations
- *(kernels)* add Triton kernels for ONNX-sourced ops and wire lowering
- add fused Conv2dBnSilu GEMM and channel-tiled kernels
- name PTX entry points after the kernel function
- fused Conv2d+BN+SiLU kernel and Graph::optimise() pass
- RuntimeOp impls for GELU/Sigmoid/LayerNorm and concrete shape propagation
- add graph node naming infrastructure for safetensors weight loading
- add CustomOp trait, lowering middleware chain, and migrate detect_decode to vision-rs
- add CustomOp trait with shape inference and lowering middleware chain
- add PSA and attention backward CUDA integration tests
- implement BatchNorm2d backward pass, PSA layout-op backward kernels, and YOLO26 op backward wiring
- add ElemwiseAdd kernel with forward+backward+RuntimeOp, replace VectorAdd
- add channel_bias_add and detect_decode kernels; fix grouped Conv2d lowering
- add MaxPool2d padding support and real RuntimeOp
- add ChannelCat/ChannelChunk RuntimeOp impls with multi-launch executor
- add UpsampleNearest2d kernel with forward/backward, graph node, and tests
- add groups to Conv2d op and Op::Attention for C2PSA graph
- *(cuda)* parse PTX metadata into KernelMetadata struct
- add supporting infrastructure for Flash Attention 2 backward
- add Flash Attention 2 backward pass kernels and tests
- add Flash Attention 2 forward kernel with CUDA test
- add ChannelChunk, ChannelCat, and Add graph ops with CUDA kernels
- add in-kernel padding to conv1d/2d/3d forward and backward kernels
- add BatchNorm training lowering with two sequential DAG nodes
- add norm kernels (GroupNorm/InstanceNorm/LayerNorm/RMSNorm/BatchNorm) and fix variance masking bug
- graph compiler with CUDA runtime and MNIST training ([#8](https://github.com/teenygrad/teenygrad/pull/8))
- wip graph compiler
- add activation kernels and fix CUDA compilation issues
- add conv2d forward/backward and avgpool2d forward/backward kernels
- add softmax forward and backward Triton kernels with tests
- add flatten_forward and flatten_backward kernels with tests
- *(teeny-triton)* add scalar comparison ops and refresh kitchen sink snapshots
- wip basic kernels with full triton integration
- wip basic kernels with full triton integration
- fully support triton ops
- Modified license to Apache 2.0 to make it fully opensource

### Fixed

- *(release)* break circular publish deps via path-only dev-dependencies
- *(macros)* document the #[kernel] macro's generated const fields and new()
- *(macros)* forward #[kernel] fn doc comments to the generated struct
- resolve all clippy warnings, tighten CI to -D warnings + fmt --check
- *(kernels)* resolve GPU execution failures discovered on RTX 5070
- Conv2dBnSiluTiled y_col_stride must be a multiple of BLOCK_OW
- use IEEE precision in Conv2dBnSilu GEMM to match cuDNN accuracy
- correct forward_output_row_stride in conv2d_bn_silu_tiled
- use NCHW-native BN kernel in training mode; add param_names to normalize op
- update batchnorm to accumulate in BLOCK_N tensors before reducing
- suppress clippy::erasing_op for I32Tensor scalar splat pattern
- include target cpu in compiler cache key to prevent parallel test races
- update softmax snapshots after softmax codegen implementation
- resolve conv2d/avgpool2d test crashes and update snapshots
- linear_backward K-tiling via 3D grid, fix with-bias test grid
- fixed incorrect triton operations (triton emits some dodgy IR which is not actually used)
- fixed some issues in the implementation of core

### Other

- apply cargo fmt --all workspace-wide
- add crate-level //! docs to every publishable crate
- prepare workspace for crates.io publishing and self-hosted docs
- remove files added to git by mistake
- centralize internal crate path+version via workspace.dependencies
- fix crates.io publish readiness for workspace manifests
- *(kernels)* ignore flaky muon_ns_xtx CUDA test on RTX 5070
- *(kernels)* add complete integration test coverage for new Triton kernel ops
- restore TF32 default in Conv2dBnSilu GEMM (revert IEEE precision)
- enable native codegen for all SM versions (sm_75–sm_120)
- update snapshots for named PTX entry points
- add numerical correctness test for fused Conv2d+BN+SiLU kernel
- remove stale PSA FA2 backward tests from teeny-kernels
- rename TEENY_CACHE_DIR to TEENYC_CACHE_DIR
- rename TEENY_RUSTC_PATH to TEENYC_PATH
- centralise workspace package metadata
- move PSA attention from Op::Attention lowering to CustomOp in vision-rs
- embed CustomOp lowering into trait, handle Op::Custom in TritonLowering
- update MLIR/source snapshots after conv padding changes
- switch batchnorm tests to PyTorch fixtures
- replace ndarray reference impl with pre-generated PyTorch fixtures
- refactor teeny-core crates
- minor refactor and wip model
- update linear MLIR snapshots for without-bias and with-bias variants
- *(teeny-kernels)* expand linear test coverage and refresh snapshots
- *(teeny-kernels)* add GPU integration tests for linear MLP kernel ([#7](https://github.com/teenygrad/teenygrad/pull/7))
- *(teeny-kernels)* add GPU integration test for relu kernel ([#6](https://github.com/teenygrad/teenygrad/pull/6))
- remove unused kernels and clean up module structure
- add rank/shape generics to dtype traits and nn layer foundations
- wip full triton dsl support
- wip relu kernel
- updated integration tests in teeny-kernels to assert of ptx generated
- modified tests and kernel name to vector add, also wip relu activation function
- updated macro to generate a kernel struct and implementation
- modified to use primitive type for f32, i32, etc which are supported in rust
- restructures
- minor name change for triton ops
- wip llvm -> triton
- wip triton llvm api
- wip triton llvm api
- wip triton llvm api
- wip triton llvm api
- wip triton llvm api
- wip triton llvm api
- wip compile basic kernel
- wip dummy triton implementation
- wip trait based dsl
- wip trait based dsl
- wip trait based dsl
- sprints now organised by quarter, sprint review and plan for new sprint
- wip trait based dsl
- wip trait based dsl
- wip new trait based dsl
- wip triton integration
- modified to use c calling convetion for kernel, and added missing panic handlers
- wip teenygrad compiler
- wip teenygrad compiler
- wip initial triton dsl
- added device to tensor type
- wip triton dsl
- rust triton dsl

### Removed

- removed old triton implementation and started work on new trait based implementation
- removed unused module and re-organised folder structure
