# teeny-llm

LLM inference application for [teenygrad](https://teenygrad.org) — vLLM-style serving, 2x faster
(see the project roadmap). Ships as a single `teeny-llm` binary.

> ⚠️ Early scaffolding: the binary currently only prints a placeholder banner. The planned
> direction is an interactive mode plus an OpenAPI-compatible HTTP server mode in one binary
> (earlier separate `agent`/`console` binaries were dropped in favor of this).

## Prerequisites

- **Rust**: any stable or nightly toolchain to `cargo build` this crate.
- Running real inference (once implemented) will need the same runtime toolchain as the rest of
  the workspace's GPU path — the custom `teenyc` compiler via `TEENYC_PATH`. See
  [`teeny-compiler`](https://docs.teenygrad.org/api/teeny-compiler/)'s README for setup.

## Getting started

```bash
cargo install teeny-llm
teeny-llm
```

## License

Apache-2.0. See [LICENSE-APACHE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE-APACHE).
