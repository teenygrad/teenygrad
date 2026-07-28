# teeny-cache

Key-value cache utilities for LLM inference in [teenygrad](https://teenygrad.org) — currently
`DynamicCache`, tracking sequence length and per-layer sliding-window/compileability state across
generation steps.

## Prerequisites

- **Rust**: any stable or nightly toolchain. No system-library or custom-compiler dependencies.

## Getting started

```toml
[dependencies]
teeny-cache = "0"
```

```rust
use teeny_cache::DynamicCache;

let cache = DynamicCache::new();
let seq_len = cache.get_sequence_length();
```

## License

Apache-2.0. See [LICENSE-APACHE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE-APACHE).
