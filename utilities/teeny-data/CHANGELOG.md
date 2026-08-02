# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0](https://github.com/teenygrad/teenygrad/releases/tag/teeny-data-v0.1.0) - 2026-08-02

### Added

- add graph node naming infrastructure for safetensors weight loading
- Modified license to Apache 2.0 to make it fully opensource

### Other

- enforce #![warn(missing_docs)] on the fully-documented crates
- document public API of the zero-coverage crates
- add crate-level //! docs to every publishable crate
- prepare workspace for crates.io publishing and self-hosted docs
- centralize internal crate path+version via workspace.dependencies
- fix crates.io publish readiness for workspace manifests
- centralise workspace package metadata
- wip convert fxgraph to egraph
- wip mixed precision graph api
- wip qwen3 embedding module
- modified to use anyhow results for better stack traces
- wip load hugging face model from safetensors
- added support for reading safetensors from the file system
- wip basic vector add
- wip model jit
- added csv data loader
- added hugging face cli
- wip download huggingface data
- added support for downloading huggingface models

### Removed

- removed teeny-hf for the moment, the code still exists on a separate branch
