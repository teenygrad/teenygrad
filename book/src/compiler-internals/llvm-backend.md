# The LLVM/MLIR Backend

`teeny-compiler::compiler::backend::llvm` compiles a kernel by shelling out to the custom
`teenyc` compiler — a fork of `rustc` with an MLIR codegen backend — as a subprocess:

```text
teenyc <kernel-source>.rs \
  -Copt-level=3 \
  -Zcodegen-backend=mlir \
  --emit=obj \
  -o <output>.o \
  --target=nvptx64-nvidia-cuda \
  --crate-type=lib \
  -C overflow-checks=off \
  --frontend=triton
```

A few details worth knowing:

- **Locating `teenyc`**: `compiler::find_teenyc` uses `TEENYC_PATH` (env var) if set, otherwise
  auto-detects the sole `rustup`-linked toolchain whose name contains `teenyc` via
  `rustup which --toolchain <name> teenyc` — see
  [Installation & Toolchain](../getting-started/installation.md). This is a *runtime* dependency
  of `teeny-compiler`, not a build-time one — `teeny-compiler` itself builds with a plain
  toolchain.
- **`-Zcodegen-backend`** is an unstable rustc flag. `teenyc` is distributed on the stable channel
  (real version numbers, normal feature-gating) — `RUSTC_BOOTSTRAP=1` is set on the subprocess
  call to permit this specific unstable flag against a stable-channel compiler, the same
  narrowly-scoped mechanism rustc's own bootstrap, `bindgen`, and `miri`'s installer rely on.
- **`--frontend=triton`** tells `teenyc` to parse the input as `teeny-triton`'s DSL rather than
  plain Rust — the kernel source file written out for compilation is the DSL text
  (`teeny_triton::triton_lang::TRITON`) concatenated with your kernel's own source.
- **Caching**: compiled kernels are cached by a hash of the kernel ID plus target
  CPU/PTX-version overrides, so repeated compiles of the same kernel/target are skipped.
- **`TEENYC_PTX_VERSION`**: overrides `teenyc`'s otherwise-conservative default PTX ISA floor for
  a given `-Ctarget-cpu` — see `teeny-compiler`'s README/rustdoc for when you need this (e.g.
  targeting a GPU architecture newer than `teenyc`'s default assumption).

> **TODO**: document the `ndarray` CPU backend as a parallel section once it has comparable
> surface area documented here.
