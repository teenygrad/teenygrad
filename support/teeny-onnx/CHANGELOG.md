# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0](https://github.com/teenygrad/teenygrad/releases/tag/teeny-onnx-v0.1.0) - 2026-07-29

### Added

- add ONNX FlexAttention/LinearAttention/CausalConvWithState support
- *(teeny-onnx)* add full ONNX operator coverage to graph IR
- *(teeny-onnx)* add public ONNX graph loader API
- Modified license to Apache 2.0 to make it fully opensource

### Fixed

- vendor the ONNX proto schema so teeny-onnx builds
- fixed incorrect copyright notice

### Other

- apply cargo fmt --all workspace-wide
- enforce #![warn(missing_docs)] on the fully-documented crates
- document public API of the zero-coverage crates
- add crate-level //! docs to every publishable crate
- prepare workspace for crates.io publishing and self-hosted docs
- remove files added to git by mistake
- centralize internal crate path+version via workspace.dependencies
- fix crates.io publish readiness for workspace manifests
- centralise workspace package metadata
- copyright change and wip implmenet tiktoken tokenizer
- added submodules again issue with submodule move

### Removed

- removed submodules
