# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/teenygrad/teenygrad/compare/teeny-cuda-v0.1.3...teeny-cuda-v0.2.0) - 2026-08-16

### Fixed

- *(cuda)* raise dynamic shared carveout; pad GEMM TMA row strides ([#33](https://github.com/teenygrad/teenygrad/pull/33))

### Other

- release v0.1.3 ([#17](https://github.com/teenygrad/teenygrad/pull/17))

## [0.1.3](https://github.com/teenygrad/teenygrad/compare/teeny-cuda-v0.1.2...teeny-cuda-v0.1.3) - 2026-08-08

### Other

- release v0.1.2 ([#16](https://github.com/teenygrad/teenygrad/pull/16))

## [0.1.2](https://github.com/teenygrad/teenygrad/compare/teeny-cuda-v0.1.1...teeny-cuda-v0.1.2) - 2026-08-06

### Other

- merge release/0.1.2 into main

## [0.1.1](https://github.com/teenygrad/teenygrad/compare/teeny-cuda-v0.1.0...teeny-cuda-v0.1.1) - 2026-08-03

### Other

- merge release/0.1.1 into main
- release v0.1.0 ([#10](https://github.com/teenygrad/teenygrad/pull/10))

## [0.1.0](https://github.com/teenygrad/teenygrad/releases/tag/teeny-cuda-v0.1.0) - 2026-08-02

### Added

- *(compiler)* auto-detect teenyc via rustup when TEENYC_PATH is unset
- add explicit ptx-version override to --options for AOT kernel compile
- add ahead-of-time kernel compilation support
- name PTX entry points after the kernel function
- *(teeny-cuda)* expose cudaProfilerStart/Stop for nsys range capture
- add CudaGraphModel::run_timed for GPU kernel timing via CUDA events
- extend CudaGraphModel to support multiple outputs
- CUDA graph capture for fixed-batch inference via LoadedModel::capture_graph
- allow TEENYC_CAPABILITY env var to override detected GPU capability
- RuntimeOp impls for GELU/Sigmoid/LayerNorm and concrete shape propagation
- trim Capability enum to sm_75+, add sm_87 and sm_120
- add graph node naming infrastructure for safetensors weight loading
- add from_host_f32 / to_host_f32 / free to TensorRef
- add terminal_node_indices_sorted_by_size to LoadedModel
- add gradient accumulation and multi-output backward to CudaModel
- add CustomOp trait, lowering middleware chain, and migrate detect_decode to vision-rs
- add ElemwiseAdd kernel with forward+backward+RuntimeOp, replace VectorAdd
- add ChannelCat/ChannelChunk RuntimeOp impls with multi-launch executor
- *(cuda)* parse PTX metadata into KernelMetadata struct
- add supporting infrastructure for Flash Attention 2 backward
- add norm kernels (GroupNorm/InstanceNorm/LayerNorm/RMSNorm/BatchNorm) and fix variance masking bug
- graph compiler with CUDA runtime and MNIST training ([#8](https://github.com/teenygrad/teenygrad/pull/8))
- wip graph compiler
- Modified license to Apache 2.0 to make it fully opensource

### Fixed

- *(release)* break circular publish deps via path-only dev-dependencies
- resolve all clippy warnings, tighten CI to -D warnings + fmt --check
- *(kernels)* resolve GPU execution failures discovered on RTX 5070
- handle cudaGetDeviceProperties API rename in CUDA 12.x aarch64
- use NCHW-native BN kernel in training mode; add param_names to normalize op
- add multi-launch support to forward_train
- expose threads_per_block/num_ctas accessors on CudaProgram
- disable bindgen comment generation for teeny-cuda to prevent doctest failures
- resolve conv2d/avgpool2d test crashes and update snapshots
- fixed incorrect copyright notice

### Other

- full public-API doc coverage for teeny-cuda
- apply cargo fmt --all workspace-wide
- add crate-level //! docs to every publishable crate
- prepare workspace for crates.io publishing and self-hosted docs
- centralize internal crate path+version via workspace.dependencies
- fix crates.io publish readiness for workspace manifests
- add pinned host memory to CudaGraphModel for faster H/D transfers
- replace per-row cuMemcpyDtoD loop with cuMemcpy2D/Async for depad
- centralise workspace package metadata
- refactor teeny-core crates
- minor refactor and wip model
- *(teeny-kernels)* add GPU integration test for relu kernel ([#6](https://github.com/teenygrad/teenygrad/pull/6))
- modified tests and kernel name to vector add, also wip relu activation function
- basic vector add kernel completed
- added cuda program launcher
- added ptx compiler to cubin
- modified compiler api and integration test
- wip integrate compilation support
- updated macro to generate a kernel struct and implementation
- added integration test to llvm compiler to ptx
- updated project to cuda toolkit 13
- wip teenygrad compiler
- wip compiler
- basic qwen3 graph conversion
- wip enhaced fxgraph schema
- modified to use anyhow results for better stack traces
- wip qwen3 cpu
- wip simple classifier
- wip compile modules
- wip optimizer and loss function
- moved hardware specific target info to appropriate crates
- wip compile graph to ndarray implementation
- wip compile graph to appropriate target
- wip basic vector add
- wip tensor ops
- wip tensor ops
- basic wiring up of devices
- wip jit macros
- wip model jit
- added support for querying for cuda devices
- wip teeny runtime
- added nvidia cuda toolkit rust bindings
- adding basic wiring up of drivers based on features
- copyright change and wip implmenet tiktoken tokenizer
- wip integrate triton
- wip build mlir tutorial
