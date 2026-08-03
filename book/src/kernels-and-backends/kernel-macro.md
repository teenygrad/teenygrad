# The `#[kernel]` Macro

[`teeny-macros`](https://docs.teenygrad.org/api/teenygrad/teeny_macros/) provides the `#[kernel]` attribute
macro (`teeny_macros::kernel`), which marks a function as a kernel definition consumed by
`teeny-triton`/`teeny-kernels`:

```rust,ignore
use teeny_macros::kernel;

#[kernel]
pub fn my_kernel(/* ... */) {
    // kernel body
}
```

See [Writing a Triton Kernel](./writing-a-kernel.md) for how a `#[kernel]`-annotated function
fits into the broader compilation pipeline.

> **TODO**: document exactly what code transformation `#[kernel]` performs
> (`macros/teeny-macros/src/macros/kernel.rs`) once it's stable enough to describe without
> immediately going stale.
