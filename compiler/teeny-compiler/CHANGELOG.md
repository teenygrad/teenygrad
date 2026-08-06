# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/teenygrad/teenygrad/compare/teeny-compiler-v0.1.0...teeny-compiler-v0.1.1) - 2026-08-03

### Other

- merge release/0.1.1 into main
- release v0.1.0 ([#10](https://github.com/teenygrad/teenygrad/pull/10))

## [0.1.0](https://github.com/teenygrad/teenygrad/releases/tag/teeny-compiler-v0.1.0) - 2026-08-02

### Added

- *(compiler)* auto-detect teenyc via rustup when TEENYC_PATH is unset
- add explicit ptx-version override to --options for AOT kernel compile
- default kernel cache dir to the deployed package's cache/ sibling
- trim Capability enum to sm_75+, add sm_87 and sm_120
- add ElemwiseAdd kernel with forward+backward+RuntimeOp, replace VectorAdd
- add norm kernels (GroupNorm/InstanceNorm/LayerNorm/RMSNorm/BatchNorm) and fix variance masking bug
- graph compiler with CUDA runtime and MNIST training ([#8](https://github.com/teenygrad/teenygrad/pull/8))
- split kernel source fields and return sha256 id as hex string
- fully support triton ops
- Modified license to Apache 2.0 to make it fully opensource
- wip mir -> triton
- update for rust 1.93.0 and disable emit llvm ir for the moment
- mlir backend moved to rustc_codegen_llvm
- upgrade to rust 1.93.0 and triton 3.6.0

### Fixed

- *(release)* break circular publish deps via path-only dev-dependencies
- resolve all clippy warnings, tighten CI to -D warnings + fmt --check
- set RUSTC_BOOTSTRAP=1 for teenyc's -Zcodegen-backend flag
- update batchnorm to accumulate in BLOCK_N tensors before reducing
- include target cpu in compiler cache key to prevent parallel test races
- resolve conv2d/avgpool2d test crashes and update snapshots
- fixed some issues in the implementation of core

### Other

- structural doc coverage for teeny-core (modules/structs/traits/enums)
- full public-API doc coverage for teeny-triton, teeny-compiler, teeny-vision
- apply cargo fmt --all workspace-wide
- accept updated teeny-compiler PTX snapshot
- add crate-level //! docs to every publishable crate
- prepare workspace for crates.io publishing and self-hosted docs
- centralize internal crate path+version via workspace.dependencies
- fix crates.io publish readiness for workspace manifests
- enable native codegen for all SM versions (sm_75–sm_120)
- update snapshots for named PTX entry points
- use teenyc as default fallback instead of rustc
- defer TEENYC_PATH check until compilation is actually needed
- rename TEENY_CACHE_DIR to TEENYC_CACHE_DIR
- rename TEENY_RUSTC_PATH to TEENYC_PATH
- centralise workspace package metadata
- refactor teeny-core crates
- minor refactor and wip model
- add rank/shape generics to dtype traits and nn layer foundations
- modified tests and kernel name to vector add, also wip relu activation function
- basic vector add kernel completed
- added cuda program launcher
- simplified interface to compiler via target specific driver
- modified compiler api and integration test
- wip integrate compilation support
- wip integrate compilation support
- updated test snapshot as parameters to kernel modified
- updated compiler to accept a kernel
- updated for custom rustc
- remove dependencies on rustc, run rustc in a sub-process
- added integration test to llvm compiler to ptx
- modified to use primitive type for f32, i32, etc which are supported in rust
- rust compiler as external dependency
- added support for F32 as type will be hard wired for F32 for the moment
- wip transform arange
- wip llvm compilation of basic kernel - relative imports
- wip integrate triton
- wip integrate triton passes
- modified to use c calling convetion for kernel, and added missing panic handlers
- wip teenygrad compiler
- wip teenygrad compiler
- wip teenygrad compiler
- wip teenygrad compiler
- wip teenygrad compiler
- wip teenygrad compiler
- wip teenygrad compiler
- wip teenygrad compiler
- wip build llvm and triton
- wip compiler
- moved fxgraph to core
- wip convert pytorch to teeny
- wip transform pytorch fxgraph to teeny fxgraph
- wip - convert pytorch fxgraph sample ops to egraph for compilation
- modified to use anyhow results for better stack traces
- wip compiler api
- wip compiler api
- added compiler project

### Removed

- removed cpu and vulkan device for the moment
- removed sprints folder as all sprint tracking moved to jira
- removed old triton implementation and started work on new trait based implementation
- removed compiler log files
- removed unused module and re-organised folder structure
