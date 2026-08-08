/*
 * Copyright (c) 2026 Teenygrad.
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

use std::fs::{File, create_dir_all};
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use derive_more::Display;
use sha2::{Digest, Sha256};
use teeny_core::compiler::{Compiler, Target};
use teeny_core::device::program::Kernel;
use tracing::info;

use crate::errors::Result;

/// `teenyc`'s own diagnostic verbosity, from least to most verbose.
///
/// Passed to `teenyc` via `RUSTC_LOG` (its standard rustc-derived logging env var), scoped to
/// just the MLIR backend's `tracing` target (`rustc_codegen_llvm::mlir`) so unrelated `rustc`
/// internals stay quiet. At `Debug`, the MLIR backend logs each pipeline stage's IR once (ttir,
/// ttgpuir, llir, llvmir, ptx/asm). At `Trace`, it additionally logs IR before/after every
/// individual MLIR pass within ttir/ttgpuir/llir — much more output.
///
/// `teenyc`'s captured stderr is relayed back through this process's own `tracing` (see
/// [`LlvmCompiler::compile`]), so it lands wherever the caller's subscriber routes `teeny_compiler`
/// events — it is not printed directly to the terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Display)]
pub enum LogLevel {
    /// `error`.
    #[display("error")]
    Error,
    /// `warn`.
    #[display("warn")]
    Warn,
    /// `info`.
    #[display("info")]
    Info,
    /// `debug`.
    #[display("debug")]
    Debug,
    /// `trace`.
    #[display("trace")]
    Trace,
}

impl std::str::FromStr for LogLevel {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "error" => Ok(Self::Error),
            "warn" => Ok(Self::Warn),
            "info" => Ok(Self::Info),
            "debug" => Ok(Self::Debug),
            "trace" => Ok(Self::Trace),
            other => Err(format!(
                "unknown log level '{other}'; expected one of error, warn, info, debug, trace"
            )),
        }
    }
}

/// `tracing` target `teenyc`'s MLIR backend logs pipeline-stage IR under; see [`LogLevel`].
const TEENYC_MLIR_LOG_TARGET: &str = "rustc_codegen_llvm::mlir";

/// Compiles kernels by shelling out to the custom `teenyc` compiler (`-Zcodegen-backend=mlir`)
/// at runtime. See [`crate::compiler::find_teenyc`] for how its path is resolved, and the crate
/// docs for the `cargo-teeny` setup this requires.
#[derive(Debug, Clone)]
pub struct LlvmCompiler {
    teenyc_path: PathBuf,
    cache_dir: PathBuf,
    target_cpu: Option<String>,
    ptx_version: Option<u32>,
    log_level: Option<LogLevel>,
}

impl LlvmCompiler {
    /// Creates a compiler that invokes the `teenyc` binary at `teenyc_path`, caching compiled
    /// kernels under `cache_dir` (created if it doesn't exist).
    ///
    /// `ptx_version` defaults from `$TEENYC_PTX_VERSION` if set (parse failures are ignored,
    /// falling back to `None`/teenyc's own default). This matters beyond the compile itself:
    /// [`Compiler::compile`]'s cache-key hash folds in `ptx_version`, so a JIT/runtime call site
    /// that never explicitly calls [`Self::with_ptx_version`] — e.g. a deployed binary just
    /// looking up an AOT-precompiled kernel cache, with no live `teenyc` to fall back to — must
    /// still agree with whatever override `cargo teeny package --options ptx-version=NN` used at
    /// AOT time, or every lookup misses. Reading the env var here means one `.env` entry (see
    /// `TEENYC_PTX_VERSION` in the deployed package's env) keeps both sides in sync without every
    /// call site having to thread the value through by hand.
    ///
    /// `log_level` similarly defaults from `$TEENYC_LOG_LEVEL` if set (parse failures are
    /// ignored, falling back to `None`) — this lets any call site, including ones that don't
    /// (or can't, e.g. a fixed helper like [`crate::compiler::driver::cuda::compile_kernel`])
    /// call [`Self::with_log_level`] directly, turn on pipeline-stage logging via the environment.
    pub fn new(teenyc_path: impl Into<PathBuf>, cache_dir: impl Into<PathBuf>) -> Result<Self> {
        let teenyc_path = teenyc_path.into();
        let cache_dir = cache_dir.into();

        if !cache_dir.exists() {
            create_dir_all(&cache_dir)?;
        }

        let ptx_version = std::env::var("TEENYC_PTX_VERSION").ok().and_then(|v| {
            v.parse().ok().or_else(|| {
                tracing::warn!(value = %v, "TEENYC_PTX_VERSION is not a valid u32; ignoring");
                None
            })
        });

        let log_level = std::env::var("TEENYC_LOG_LEVEL").ok().and_then(|v| {
            v.parse().ok().or_else(|| {
                tracing::warn!(value = %v, "TEENYC_LOG_LEVEL is not a valid log level; ignoring");
                None
            })
        });

        Ok(Self {
            teenyc_path,
            cache_dir,
            target_cpu: None,
            ptx_version,
            log_level,
        })
    }

    /// Sets the target GPU architecture (e.g. `sm_90`) passed to `teenyc` as `-Ctarget-cpu`.
    pub fn with_target_cpu(mut self, cpu: impl Into<String>) -> Self {
        self.target_cpu = Some(cpu.into());
        self
    }

    /// Override the PTX ISA version `teenyc` stamps into the generated PTX
    /// (encoded as `major*10 + minor`, e.g. `82` for `8.2`), via
    /// `TEENYC_PTX_VERSION`. Without this, `teenyc` picks a conservative
    /// default from the target capability — set this when the deployment
    /// target's exact CUDA version is known and needs a precise match.
    pub fn with_ptx_version(mut self, ptx_version: u32) -> Self {
        self.ptx_version = Some(ptx_version);
        self
    }

    /// Sets `teenyc`'s diagnostic verbosity (see [`LogLevel`]). Left unset (the default),
    /// `teenyc` uses its own default (roughly `warn`) and no pipeline-stage IR is captured.
    pub fn with_log_level(mut self, log_level: LogLevel) -> Self {
        self.log_level = Some(log_level);
        self
    }
}

impl Compiler for LlvmCompiler {
    fn compile(&self, kernel: &impl Kernel, _target: &impl Target, force: bool) -> Result<String> {
        // Hash the kernel id together with target cpu and ptx version so that
        // different targets/overrides each get their own cache entry.
        let effective_id = {
            let mut h = Sha256::new();
            h.update(kernel.id().as_bytes());
            if let Some(cpu) = &self.target_cpu {
                h.update(cpu.as_bytes());
            }
            if let Some(ptx_version) = self.ptx_version {
                h.update(ptx_version.to_le_bytes());
            }
            h.finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        };
        let kernel_file_name = format!("{}_{}", kernel.name(), effective_id);
        let kernel_file = self.cache_dir.join(&kernel_file_name).with_extension("rs");
        let output_file = self
            .cache_dir
            .join(kernel_file_name.clone())
            .with_extension("o");

        if !output_file.exists() || force {
            anyhow::ensure!(
                self.teenyc_path.exists(),
                "kernel not cached and rustc not found at {:?}; \
                 set TEENYC_PATH to a valid rustc binary",
                self.teenyc_path
            );

            // Two callers compiling the *same* kernel hash concurrently (e.g. under `cargo
            // test`'s default parallelism) must not both write `kernel_file`/invoke `teenyc`
            // at once -- that races on the shared source path and can corrupt the output.
            // A lock file scoped to this exact hash serializes only that collision: unrelated
            // hashes use different lock files and keep compiling fully in parallel. The lock
            // file itself is intentionally never deleted -- unlinking it here would race a
            // concurrent locker into flock'ing a since-replaced inode, defeating the lock.
            let lock_path = self.cache_dir.join(format!("{kernel_file_name}.lock"));
            let mut lock = fd_lock::RwLock::new(File::create(&lock_path)?);
            let _guard = lock.write()?;

            // Double-check after acquiring the lock: if another process compiled (and
            // released the lock for) this exact hash while we were waiting, reuse its
            // output rather than redundantly recompiling. This -- not just avoiding the
            // corrupted-write symptom -- is the actual point of taking the lock.
            if !output_file.exists() || force {
                let mut file = File::create(&kernel_file)?;

                info!("Writing kernel code to file");
                file.write_all(teeny_triton::triton_lang::TRITON.as_bytes())?;
                file.write_all(kernel.source().as_bytes())?;

                // Compile to a unique per-process temp path and atomically rename it into
                // place afterwards (POSIX `rename` is atomic within the same directory), so
                // any reader of `output_file` never observes a partial write.
                let tmp_output_file = self
                    .cache_dir
                    .join(format!("{kernel_file_name}.o.tmp.{}", std::process::id()));

                let mut cmd = Command::new(&self.teenyc_path);
                cmd.arg(&kernel_file)
                    .arg("-Copt-level=3")
                    .arg("-Zcodegen-backend=mlir")
                    .arg("--emit=obj")
                    .arg(format!("-o{}", tmp_output_file.display()))
                    .arg("--target=nvptx64-nvidia-cuda")
                    .arg("--crate-type=lib")
                    .arg("-C")
                    .arg("overflow-checks=off")
                    .arg("--frontend=triton")
                    .current_dir(&self.cache_dir)
                    // `-Zcodegen-backend` is an unstable flag; `teenyc` is distributed on the
                    // "stable" channel (real version numbers, normal feature-gating), so without
                    // this it refuses with "the option `Z` is only accepted on the nightly
                    // compiler". `RUSTC_BOOTSTRAP=1` is the standard, narrowly-scoped way to permit
                    // specific unstable flags against a stable-channel compiler (the same mechanism
                    // rustc's own bootstrap, bindgen, and miri's installer rely on) without needing
                    // to distribute `teenyc` itself as a nightly build.
                    .env("RUSTC_BOOTSTRAP", "1");
                if let Some(cpu) = &self.target_cpu {
                    cmd.arg(format!("-Ctarget-cpu={cpu}"));
                }
                if let Some(ptx_version) = self.ptx_version {
                    cmd.env("TEENYC_PTX_VERSION", ptx_version.to_string());
                }
                if let Some(log_level) = self.log_level {
                    cmd.env("RUSTC_LOG", format!("{TEENYC_MLIR_LOG_TARGET}={log_level}"));
                }
                let output = cmd.output()?;

                // `teenyc`'s own tracing subscriber writes to its stderr; relay it through this
                // process's tracing rather than printing directly, so it's filterable/routable like
                // any other event here. Only worth the string conversion when logging was requested.
                if self.log_level.is_some() {
                    for line in String::from_utf8_lossy(&output.stderr).lines() {
                        tracing::debug!(target: "teeny_compiler::llvm::teenyc", "{line}");
                    }
                }

                if !output.status.success() {
                    let _ = std::fs::remove_file(&tmp_output_file);
                    let _ = std::fs::remove_file(tmp_output_file.with_extension("mlir"));
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    anyhow::bail!("rustc exited with status {}\n{}", output.status, stderr);
                }

                // `teenyc` also writes a `.mlir` sidecar next to the object file (the
                // pre-Triton MLIR source, read back by e.g. the `*_mlir_output` snapshot
                // tests) -- derived from the same `-o` path, so it needs the same
                // temp-then-rename treatment. It may not exist for every invocation, hence
                // the existence check rather than treating a missing file as an error.
                let tmp_mlir_file = tmp_output_file.with_extension("mlir");
                if tmp_mlir_file.exists() {
                    std::fs::rename(&tmp_mlir_file, output_file.with_extension("mlir"))?;
                }

                std::fs::rename(&tmp_output_file, &output_file)?;
            }
        }

        Ok(output_file.to_string_lossy().to_string())
    }
}
