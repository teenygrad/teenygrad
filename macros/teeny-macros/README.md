# teeny-macros

Procedural macros for [teenygrad](https://teenygrad.org). Currently provides the `#[kernel]`
attribute macro, used to mark functions as GPU/CPU kernel definitions consumed by
`teeny-triton`/`teeny-kernels`.

## Prerequisites

- **Rust**: any stable or nightly toolchain. Pure proc-macro crate (`syn`/`quote`/`proc-macro2`),
  no system-library or custom-compiler dependencies.

## Getting started

```toml
[dependencies]
teeny-macros = "0"
```

```rust
use teeny_macros::kernel;

#[kernel]
pub fn my_kernel(/* ... */) {
    // kernel body
}
```

`teeny-macros` is a building block for `teeny-kernels`; most users won't depend on it directly.

## License

Apache-2.0. See [LICENSE-APACHE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE-APACHE).
