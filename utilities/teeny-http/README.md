# teeny-http

Small HTTP download/fetch helpers for [teenygrad](https://teenygrad.org), built on `reqwest` with
`indicatif` progress reporting — used for pulling models/datasets/assets.

## Prerequisites

- **Rust**: any stable or nightly toolchain. No system-library or custom-compiler dependencies
  (TLS comes from `reqwest`'s own dependency stack).

## Getting started

```toml
[dependencies]
teeny-http = "0"
```

See the `download` and `fetch` modules for the available helpers.

## License

Apache-2.0. See [LICENSE-APACHE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE-APACHE).
