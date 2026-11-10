/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! Executes a compiled RISC-V kernel `.so` under `qemu-riscv64`.
//!
//! There is no Rust toolchain support assumed here: rather than cross-compiling the whole test
//! binary for `riscv64gc-unknown-linux-gnu` and running *that* under emulation (a much larger
//! lift -- the entire dependency graph would need to cross-compile, and kernel compilation
//! itself, which shells out to `teenyc`/LLVM, has to stay on the host regardless), this uses a
//! tiny standalone C harness, cross-compiled once and cached, that `dlopen`s the kernel `.so`,
//! calls its entry point once per block over buffers read from files, and writes the result
//! back. The host-native test process just orchestrates: compile the kernel (host-side),
//! compile the harness (once, cached), then run the harness under `qemu-riscv64`.

use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, bail, ensure};

use super::resolve_host_tools;
use crate::teenyc_cache_dir;

/// `dlopen`s argv[1] and runs the pointwise kernel named by argv[2] over the f32 buffers in
/// argv[5] (`x`) and argv[6] (`y`), one call per argv[3]-element block with argv[4] as
/// `n_elements`, then writes `y` back to argv[6]. Exits 0 on success; prints a diagnostic and
/// exits non-zero on any failure.
///
/// The kernel's C signature is the Triton entry point's own `(x_ptr, y_ptr, n_elements)`
/// followed by the six program-context arguments TritonCPU appends when lowering it: the program
/// id and then the grid size, each for axes x, y and z.
const HARNESS_SOURCE: &str = r#"
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef void (*pointwise_kernel)(float *x, float *y, int32_t n_elements,
                                 int32_t pid_x, int32_t pid_y, int32_t pid_z,
                                 uint32_t grid_x, uint32_t grid_y, uint32_t grid_z);

static float *read_floats(const char *path, size_t *count) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    rewind(f);
    if (bytes < 0 || bytes % sizeof(float) != 0) {
        fprintf(stderr, "%s: size %ld is not a whole number of f32 values\n", path, bytes);
        fclose(f);
        return NULL;
    }
    *count = (size_t)bytes / sizeof(float);
    float *data = malloc(bytes > 0 ? (size_t)bytes : 1);
    if (!data || fread(data, sizeof(float), *count, f) != *count) {
        fprintf(stderr, "%s: failed to read %zu f32 values\n", path, *count);
        free(data);
        fclose(f);
        return NULL;
    }
    fclose(f);
    return data;
}

int main(int argc, char **argv) {
    if (argc != 7) {
        fprintf(stderr,
                "usage: %s <shared-library> <symbol> <block-size> <n-elements> <x.bin> <y.bin>\n",
                argv[0]);
        return 2;
    }
    void *handle = dlopen(argv[1], RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "dlopen(%s) failed: %s\n", argv[1], dlerror());
        return 1;
    }
    pointwise_kernel kernel = (pointwise_kernel)dlsym(handle, argv[2]);
    if (!kernel) {
        fprintf(stderr, "dlsym(%s) failed: %s\n", argv[2], dlerror());
        return 1;
    }

    long block_size = atol(argv[3]);
    long n_elements = atol(argv[4]);
    size_t x_count = 0, y_count = 0;
    float *x = read_floats(argv[5], &x_count);
    float *y = read_floats(argv[6], &y_count);
    if (!x || !y) {
        return 1;
    }
    if (block_size <= 0 || x_count != y_count || x_count % (size_t)block_size != 0 ||
        n_elements < 0 || (size_t)n_elements > x_count) {
        fprintf(stderr,
                "invalid buffers: block size %ld, n_elements %ld, x has %zu, y has %zu\n",
                block_size, n_elements, x_count, y_count);
        return 2;
    }

    uint32_t blocks = (uint32_t)(x_count / (size_t)block_size);
    for (uint32_t pid = 0; pid < blocks; ++pid) {
        kernel(x, y, (int32_t)n_elements, (int32_t)pid, 0, 0, blocks, 1, 1);
    }

    FILE *out = fopen(argv[6], "wb");
    if (!out || fwrite(y, sizeof(float), y_count, out) != y_count || fclose(out) != 0) {
        fprintf(stderr, "%s: failed to write the kernel's output\n", argv[6]);
        return 1;
    }
    return 0;
}
"#;

/// A RISC-V + QEMU test environment: the resolved cross C compiler / `qemu-riscv64`, the cross
/// toolchain's sysroot (needed by `qemu-riscv64 -L` to find the RISC-V dynamic linker/libc,
/// since the harness below `dlopen`s and so can't be statically linked), and the compiled
/// dlopen/dlsym harness binary.
pub struct QemuTestEnv {
    qemu: String,
    sysroot: PathBuf,
    harness: PathBuf,
}

/// Resolves the RISC-V cross toolchain and `qemu-riscv64` (see [`super::RiscvHostTools`]),
/// then compiles (or reuses a cached build of) the dlopen/dlsym harness. Errors clearly if a
/// required tool is missing -- callers should let this test fail loudly rather than skip.
pub fn setup_qemu_env() -> Result<QemuTestEnv> {
    let tools = resolve_host_tools()?;
    let sysroot = cc_sysroot(&tools.cc)?;
    let harness = compile_harness(&tools.cc)?;

    Ok(QemuTestEnv {
        qemu: tools.qemu,
        sysroot,
        harness,
    })
}

impl QemuTestEnv {
    /// Runs the pointwise kernel `symbol` from `so_path` under `qemu-riscv64`, calling it once
    /// per `block_size`-element block of `x` with `n_elements`, as a Triton launcher would.
    ///
    /// The kernel must have the Triton signature `(x_ptr, y_ptr, n_elements)`. `x` and `y` must
    /// be the same length, a whole number of blocks, since the kernel's masked loads and stores
    /// address whole blocks. `y` is overwritten with the buffer as the kernel left it, including
    /// the padding past `n_elements`, so callers can check the mask as well as the results.
    pub fn run_pointwise_kernel(
        &self,
        so_path: &Path,
        symbol: &str,
        block_size: usize,
        n_elements: usize,
        x: &[f32],
        y: &mut [f32],
    ) -> Result<()> {
        ensure!(block_size > 0, "block_size must be positive");
        ensure!(
            x.len() == y.len() && x.len().is_multiple_of(block_size) && n_elements <= x.len(),
            "x ({}) and y ({}) must be the same whole number of {block_size}-element blocks \
             covering n_elements ({n_elements})",
            x.len(),
            y.len(),
        );

        static NEXT_RUN: AtomicUsize = AtomicUsize::new(0);
        let scratch = PathBuf::from(teenyc_cache_dir());
        std::fs::create_dir_all(&scratch)
            .with_context(|| format!("failed to create cache dir {}", scratch.display()))?;
        let stem = format!(
            "riscv_qemu_run_{}_{}",
            std::process::id(),
            NEXT_RUN.fetch_add(1, Ordering::Relaxed)
        );
        let x_path = scratch.join(format!("{stem}_x.bin"));
        let y_path = scratch.join(format!("{stem}_y.bin"));

        let result = self.run_harness(
            so_path, symbol, block_size, n_elements, x, y, &x_path, &y_path,
        );
        let _ = std::fs::remove_file(&x_path);
        let _ = std::fs::remove_file(&y_path);
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn run_harness(
        &self,
        so_path: &Path,
        symbol: &str,
        block_size: usize,
        n_elements: usize,
        x: &[f32],
        y: &mut [f32],
        x_path: &Path,
        y_path: &Path,
    ) -> Result<()> {
        write_floats(x_path, x)?;
        write_floats(y_path, y)?;

        let output = Command::new(&self.qemu)
            // The kernels use RVV instructions; `max` enables every extension QEMU emulates.
            .args(["-cpu", "max"])
            .arg("-L")
            .arg(&self.sysroot)
            .arg(&self.harness)
            .arg(so_path)
            .arg(symbol)
            .arg(block_size.to_string())
            .arg(n_elements.to_string())
            .arg(x_path)
            .arg(y_path)
            .output()
            .with_context(|| {
                format!(
                    "failed to run {} under qemu-riscv64",
                    self.harness.display()
                )
            })?;

        if !output.status.success() {
            bail!(
                "qemu-riscv64 harness exited with {}: stdout={} stderr={}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }

        let result = read_floats(y_path)?;
        ensure!(
            result.len() == y.len(),
            "harness wrote {} f32 values, expected {}",
            result.len(),
            y.len()
        );
        y.copy_from_slice(&result);
        Ok(())
    }
}

/// Writes `values` as little-endian f32s, the byte order of both the host and RISC-V.
fn write_floats(path: &Path, values: &[f32]) -> Result<()> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))
}

fn read_floats(path: &Path) -> Result<Vec<f32>> {
    let bytes =
        std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    ensure!(
        bytes.len() % 4 == 0,
        "{} is not a whole number of f32 values",
        path.display()
    );
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// Debian/Ubuntu-packaged cross toolchains (`gcc-riscv64-linux-gnu`) install their target libc
/// under `/usr/<multiarch-triplet>` and don't wire `-print-sysroot` to point there (it just
/// prints `/`, the host root) -- `-print-multiarch` is what's actually reliable for this
/// toolchain layout. Fall back to `-print-sysroot` for a toolchain that *does* support it
/// (e.g. a from-source crosstool-ng build), erroring clearly if neither yields a real directory.
fn cc_sysroot(cc: &str) -> Result<PathBuf> {
    let multiarch = run_cc(cc, "-print-multiarch")?;
    if !multiarch.is_empty() {
        let sysroot = PathBuf::from("/usr").join(&multiarch);
        if sysroot.is_dir() {
            return Ok(sysroot);
        }
    }

    let sysroot = run_cc(cc, "-print-sysroot")?;
    if !sysroot.is_empty() && sysroot != "/" && Path::new(&sysroot).is_dir() {
        return Ok(PathBuf::from(sysroot));
    }

    bail!(
        "could not determine a RISC-V sysroot for `{cc}` (checked `/usr/{multiarch}` and \
         `-print-sysroot`); install the matching cross-libc package (e.g. \
         `libc6-dev-riscv64-cross` alongside `gcc-riscv64-linux-gnu`)"
    );
}

fn run_cc(cc: &str, arg: &str) -> Result<String> {
    let output = Command::new(cc)
        .arg(arg)
        .output()
        .with_context(|| format!("failed to run `{cc} {arg}`"))?;
    if !output.status.success() {
        bail!(
            "`{cc} {arg}` exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Compiles [`HARNESS_SOURCE`] once into `teenyc_cache_dir()`, keyed by a hash of the source so
/// a future change to the embedded harness invalidates any binary a prior run left cached.
fn compile_harness(cc: &str) -> Result<PathBuf> {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    HARNESS_SOURCE.hash(&mut hasher);
    let cache_dir = PathBuf::from(teenyc_cache_dir());
    let bin_path = cache_dir.join(format!("riscv_qemu_harness_{:x}", hasher.finish()));

    if bin_path.is_file() {
        return Ok(bin_path);
    }

    std::fs::create_dir_all(&cache_dir)
        .with_context(|| format!("failed to create cache dir {}", cache_dir.display()))?;

    let src_path = cache_dir.join("riscv_qemu_harness.c");
    std::fs::write(&src_path, HARNESS_SOURCE)
        .with_context(|| format!("failed to write harness source to {}", src_path.display()))?;

    let output = Command::new(cc)
        .args(["-O2", "-o"])
        .arg(&bin_path)
        .arg(&src_path)
        .arg("-ldl")
        .output()
        .with_context(|| format!("failed to run {cc} to compile the QEMU dlopen harness"))?;

    if !output.status.success() {
        bail!(
            "{cc} failed to compile the QEMU dlopen harness: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    Ok(bin_path)
}
