# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1](https://github.com/teenygrad/teenygrad/compare/teeny-cli-v0.1.0...teeny-cli-v0.1.1) - 2026-08-02

### Other

- release v0.1.0 ([#10](https://github.com/teenygrad/teenygrad/pull/10))

## [0.1.0](https://github.com/teenygrad/teenygrad/releases/tag/teeny-cli-v0.1.0) - 2026-08-02

### Added

- *(compiler)* auto-detect teenyc via rustup when TEENYC_PATH is unset
- add ahead-of-time kernel compilation support

### Other

- apply cargo fmt --all workspace-wide
- enforce #![warn(missing_docs)] on the fully-documented crates
- prepare workspace for crates.io publishing and self-hosted docs
