# Contributing to Teenygrad

Thank you for your interest in contributing! The full, authoritative guide lives in the
repository's [`CONTRIBUTING.md`](https://github.com/teenygrad/teenygrad/blob/main/CONTRIBUTING.md)
— this chapter summarizes the essentials.

We follow the contributor model used by Ubuntu, and require a signed
[CLA](https://www.teenygrad.org/contributing) before accepting contributions. Reach out on
[Discord](https://discord.gg/FG65XyPzuD) with any questions.

All contributions require:

- **An issue** first — file a feature request or bug report to get a maintainer's attention and
  find the right project for it.
- **A pull request**, approved by the appropriate reviewers.

## Engineering standards

See the repository's `AGENTS.md` and `CLAUDE.md` for the full Rust engineering standards
(idiomatic Rust, error handling, `unsafe` documentation requirements, testing expectations). In
short: prefer clarity over cleverness, minimize cloning/allocation, avoid `unwrap()`/`expect()` in
non-test code, and add tests for behavior changes.

## Documentation standards

- Every publishable crate should have a crate-level `//!` doc comment (see
  [Introduction](../introduction.md) for how this book fits alongside per-crate rustdoc) and
  `///` doc comments on public items.
- New crates need a crate-specific `README.md` (not the shared workspace one) — see any existing
  crate's `README.md` for the expected shape (a short description, a "Prerequisites" section
  covering any system dependencies/toolchain requirements, and a minimal usage example).
