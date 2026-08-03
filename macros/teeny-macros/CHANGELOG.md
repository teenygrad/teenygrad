# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/teenygrad/teenygrad/compare/teeny-macros-v0.1.0...teeny-macros-v0.1.1) - 2026-08-03

### Other

- merge release/0.1.1 into main
- release v0.1.0 ([#10](https://github.com/teenygrad/teenygrad/pull/10))

## [0.1.0](https://github.com/teenygrad/teenygrad/releases/tag/teeny-macros-v0.1.0) - 2026-08-02

### Added

- add dtype-aware kernel dispatch and generic float activations
- name PTX entry points after the kernel function
- graph compiler with CUDA runtime and MNIST training ([#8](https://github.com/teenygrad/teenygrad/pull/8))
- wip basic kernels with full triton integration
- split kernel source fields and return sha256 id as hex string
- Modified license to Apache 2.0 to make it fully opensource

### Fixed

- *(macros)* document the #[kernel] macro's generated const fields and new()
- *(macros)* forward #[kernel] fn doc comments to the generated struct
- resolve all clippy warnings, tighten CI to -D warnings + fmt --check

### Other

- full public-API doc coverage for teeny-triton, teeny-compiler, teeny-vision
- apply cargo fmt --all workspace-wide
- add crate-level //! docs to every publishable crate
- prepare workspace for crates.io publishing and self-hosted docs
- remove files added to git by mistake
- centralise workspace package metadata
- refactor teeny-core crates
- wip integrate compilation support
- updated macro to generate a kernel struct and implementation
- wip llvm compilation of basic kernel - relative imports
- modified to use c calling convetion for kernel, and added missing panic handlers
- wip teenygrad compiler
- wip compiler
- wip triton dsl
- rust triton dsl

### Removed

- removed unused module and re-organised folder structure
