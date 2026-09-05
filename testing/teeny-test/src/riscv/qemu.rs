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
//! itself, which shells out to `teenyc`/LLVM, has to stay on the host regardless), this mirrors
//! the approach already verified by hand for `teeny-riscv`: a tiny standalone C harness,
//! cross-compiled once and cached, that `dlopen`s the kernel `.so` and calls a named
//! no-argument symbol. The host-native test process just orchestrates: compile the kernel
//! (host-side), compile the harness (once, cached), then run the harness under `qemu-riscv64`.

use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

use super::resolve_host_tools;
use crate::teenyc_cache_dir;

/// `dlopen`s argv[1] and calls the no-argument, no-return symbol named by argv[2]. Exits 0 on
/// success; prints a diagnostic and exits non-zero on any dlopen/dlsym failure.
const HARNESS_SOURCE: &str = r#"
#include <dlfcn.h>
#include <stdio.h>

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <shared-library> <symbol>\n", argv[0]);
        return 2;
    }
    void *handle = dlopen(argv[1], RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "dlopen(%s) failed: %s\n", argv[1], dlerror());
        return 1;
    }
    void (*kernel)(void) = (void (*)(void))dlsym(handle, argv[2]);
    if (!kernel) {
        fprintf(stderr, "dlsym(%s) failed: %s\n", argv[2], dlerror());
        return 1;
    }
    kernel();
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
    /// Runs the harness under `qemu-riscv64`, `dlopen`ing `so_path` and calling `symbol` as a
    /// no-argument, no-return function. Returns an error (including the harness's stderr) if
    /// the process exits non-zero.
    pub fn run_kernel(&self, so_path: &Path, symbol: &str) -> Result<()> {
        let output = Command::new(&self.qemu)
            .arg("-L")
            .arg(&self.sysroot)
            .arg(&self.harness)
            .arg(so_path)
            .arg(symbol)
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

        Ok(())
    }
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
