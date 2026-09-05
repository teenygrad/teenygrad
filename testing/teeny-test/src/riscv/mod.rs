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

//! RISC-V test support. Compiling a kernel for RISC-V (`teeny_riscv::compiler::compile_kernel`)
//! works on any host, since it only shells out to `teenyc`/LLVM. *Running* the compiled `.so`
//! needs RISC-V hardware or user-mode emulation -- see [`qemu`] (gated behind the separate
//! `qemu` feature, since it needs `qemu-riscv64` and a `riscv64-linux-gnu` cross toolchain on
//! the host, not just this crate's `riscv` feature).

use anyhow::{Context, Result, bail};

/// RISC-V-related host tools resolved for a test run: a cross C compiler for
/// `riscv64-linux-gnu` and `qemu-riscv64` for user-mode emulation.
///
/// Both default to their bare name on `PATH` and are overridable via the `TEENYC_RISCV_CC` /
/// `TEENYC_QEMU_RISCV64` env vars (e.g. for a toolchain installed under a non-standard prefix).
/// Resolution fails clearly (rather than skipping the test silently) when a tool the caller
/// needs isn't found -- see [`qemu::setup_qemu_env`], the only current caller.
pub struct RiscvHostTools {
    /// Path (or bare name) of the `riscv64-linux-gnu-gcc`-equivalent cross C compiler.
    pub cc: String,
    /// Path (or bare name) of `qemu-riscv64`.
    pub qemu: String,
}

/// Resolves [`RiscvHostTools`], erroring with a clear, actionable message if a requested tool
/// can't be found on `PATH` (via `which`) or the overriding env var doesn't point to a real file.
pub(crate) fn resolve_host_tools() -> Result<RiscvHostTools> {
    let cc = resolve_tool("TEENYC_RISCV_CC", "riscv64-linux-gnu-gcc")?;
    let qemu = resolve_tool("TEENYC_QEMU_RISCV64", "qemu-riscv64")?;
    Ok(RiscvHostTools { cc, qemu })
}

fn resolve_tool(env_var: &str, default_name: &str) -> Result<String> {
    if let Ok(path) = std::env::var(env_var) {
        if !std::path::Path::new(&path).is_file() {
            bail!("{env_var}={path:?} does not point to an existing file");
        }
        return Ok(path);
    }

    which_on_path(default_name).with_context(|| {
        format!(
            "'{default_name}' not found on PATH -- install it (e.g. `apt install \
             gcc-riscv64-linux-gnu qemu-user`), or point {env_var} at it directly"
        )
    })
}

fn which_on_path(name: &str) -> Result<String> {
    let path_var = std::env::var_os("PATH").unwrap_or_default();
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Ok(candidate.to_string_lossy().into_owned());
        }
    }
    bail!("'{name}' not found in any PATH directory")
}

#[cfg(feature = "qemu")]
/// Compiling the placeholder kernel's `.so` to a real host executable that `dlopen`s and calls
/// it, run under `qemu-riscv64`.
pub mod qemu;
