# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2](https://github.com/teenygrad/teenygrad/compare/teeny-triton-v0.1.1...teeny-triton-v0.1.2) - 2026-08-06

### Other

- merge release/0.1.2 into main

## [0.1.1](https://github.com/teenygrad/teenygrad/compare/teeny-triton-v0.1.0...teeny-triton-v0.1.1) - 2026-08-03

### Other

- merge release/0.1.1 into main
- release v0.1.0 ([#10](https://github.com/teenygrad/teenygrad/pull/10))

## [0.1.0](https://github.com/teenygrad/teenygrad/releases/tag/teeny-triton-v0.1.0) - 2026-08-02

### Added

- *(compiler)* auto-detect teenyc via rustup when TEENYC_PATH is unset
- make flash_attn2 generic over Float
- add atan to Triton DSL and LlvmTriton stub
- add supporting infrastructure for Flash Attention 2 backward
- add norm kernels (GroupNorm/InstanceNorm/LayerNorm/RMSNorm/BatchNorm) and fix variance masking bug
- graph compiler with CUDA runtime and MNIST training ([#8](https://github.com/teenygrad/teenygrad/pull/8))
- add activation kernels and fix CUDA compilation issues
- *(teeny-triton)* add scalar comparison ops and refresh kitchen sink snapshots
- wip basic kernels with full triton integration
- wip basic kernels with full triton integration
- kitchen sink test for the triton api compiler
- fully support triton ops
- Modified license to Apache 2.0 to make it fully opensource
- update for rust 1.93.0 and disable emit llvm ir for the moment

### Fixed

- *(release)* make remaining internal dev-dependencies path-only
- *(release)* break circular publish deps via path-only dev-dependencies
- resolve all clippy warnings, tighten CI to -D warnings + fmt --check
- resolve conv2d/avgpool2d test crashes and update snapshots
- add Neg and usize impls to no-core shim; use const generic for boundary_check
- fixed incorrect triton operations (triton emits some dodgy IR which is not actually used)
- fixed some issues in the implementation of core

### Other

- full public-API doc coverage for teeny-triton, teeny-compiler, teeny-vision
- apply cargo fmt --all workspace-wide
- add crate-level //! docs to every publishable crate
- prepare workspace for crates.io publishing and self-hosted docs
- centralize internal crate path+version via workspace.dependencies
- fix crates.io publish readiness for workspace manifests
- Add Triton::load_scalar_f32_as_i32 for scalar gather
- rename TEENY_CACHE_DIR to TEENYC_CACHE_DIR
- rename TEENY_RUSTC_PATH to TEENYC_PATH
- centralise workspace package metadata
- update MLIR/source snapshots after conv padding changes
- refactor teeny-core crates
- minor refactor and wip model
- add rank/shape generics to dtype traits and nn layer foundations
- wip full triton dsl support
- modified tests and kernel name to vector add, also wip relu activation function
- updated compiler to accept a kernel
- modified to use primitive type for f32, i32, etc which are supported in rust
- restructures
- added support for F32 as type will be hard wired for F32 for the moment
- minor name change for triton ops
- wip llvm -> triton
- wip llvm -> triton
- wip llvm to triton
- wip transform arange
- added basic arithmetic ops to core
- basic compilation to mlir module completed
- wip llvm compilation of basic kernel - relative imports
- wip llvm compilation of basic kernel - relative imports
- wip llvm compilation of basic kernel
- wip triton llvm api
- wip triton llvm api
- wip triton llvm api
- wip triton llvm api
- wip triton llvm api
- wip triton llvm api
- wip triton llvm api
- wip triton llvm api
- wip triton llvm api
- wip triton llvm api
- wip triton llvm api
- wip triton llvm api
- wip llvm triton api
- sprint review and plan for next sprint
- wip dummy triton implementation
- wip trait based dsl
- wip trait based dsl
- wip trait based dsl
- wip trait based dsl
- wip trait based triton dsl
- wip trait based dsl
- wip new trait based dsl
- wip integrate triton
- wip triton integration
- wip integrate triton passes
- modified to use c calling convetion for kernel, and added missing panic handlers
- wip teenygrad compiler
- wip teenygrad compiler
- wip teenygrad compiler
- wip compiler
- wip initial triton dsl
- rust triton dsl
- wip z3 based type system

### Removed

- removed old triton implementation and started work on new trait based implementation
