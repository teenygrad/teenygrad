# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2](https://github.com/teenygrad/teenygrad/compare/teeny-vision-v0.1.1...teeny-vision-v0.1.2) - 2026-08-06

### Other

- merge release/0.1.2 into main

## [0.1.1](https://github.com/teenygrad/teenygrad/compare/teeny-vision-v0.1.0...teeny-vision-v0.1.1) - 2026-08-03

### Other

- merge release/0.1.1 into main
- release v0.1.0 ([#10](https://github.com/teenygrad/teenygrad/pull/10))

## [0.1.0](https://github.com/teenygrad/teenygrad/releases/tag/teeny-vision-v0.1.0) - 2026-08-02

### Added

- *(compiler)* auto-detect teenyc via rustup when TEENYC_PATH is unset
- switch MNIST example from MLP to LeNet-5
- *(cuda)* parse PTX metadata into KernelMetadata struct
- add norm kernels (GroupNorm/InstanceNorm/LayerNorm/RMSNorm/BatchNorm) and fix variance masking bug
- graph compiler with CUDA runtime and MNIST training ([#8](https://github.com/teenygrad/teenygrad/pull/8))
- modified mnist to be generic on float type and use type inference
- moved mnist model to teeny-vision

### Fixed

- *(release)* make remaining internal dev-dependencies path-only
- expose threads_per_block/num_ctas accessors on CudaProgram

### Other

- full public-API doc coverage for teeny-triton, teeny-compiler, teeny-vision
- apply cargo fmt --all workspace-wide
- add crate-level //! docs to every publishable crate
- prepare workspace for crates.io publishing and self-hosted docs
- centralize internal crate path+version via workspace.dependencies
- fix crates.io publish readiness for workspace manifests
- rename TEENY_CACHE_DIR to TEENYC_CACHE_DIR
- rename TEENY_RUSTC_PATH to TEENYC_PATH
- centralise workspace package metadata
- remove unused kernels and clean up module structure
- restructures
