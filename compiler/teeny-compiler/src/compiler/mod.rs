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

use std::path::PathBuf;
use std::process::Command;

use anyhow::Context;

use crate::errors::Result;

/// Compilation backends (LLVM/MLIR, `ndarray`).
pub mod backend;

/// Resolve the `teenyc` binary to invoke.
///
/// Priority:
/// 1. `$TEENYC_PATH`, if set.
/// 2. The sole `rustup`-linked toolchain whose name contains `teenyc` — the naming convention
///    `cargo teeny install-toolchain` uses (default toolchain name is `<channel>-<host>` with
///    `channel` defaulting to `stable-teenyc`; see `cargo-teeny`'s `install_toolchain` module) —
///    resolved to a binary path via `rustup which --toolchain <name> teenyc`.
///
/// This deliberately does not fall back further to a bare `teenyc` looked up on `$PATH`: that
/// would only work by accident (most `$PATH`s don't have a `teenyc` on them at all) and produces
/// a worse error than pointing at the two supported setup paths.
pub fn find_teenyc() -> Result<PathBuf> {
    if let Ok(path) = std::env::var("TEENYC_PATH") {
        return Ok(PathBuf::from(path));
    }

    let toolchain = sole_teenyc_toolchain()?;
    which_in_toolchain(&toolchain)
}

/// Names of installed `rustup` toolchains containing `teenyc`, parsed from `rustup toolchain
/// list` output (one name per line, optionally suffixed with ` (default)`/` (active)`).
fn teenyc_toolchain_names(rustup_toolchain_list_output: &str) -> Vec<String> {
    rustup_toolchain_list_output
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .filter(|name| name.contains("teenyc"))
        .map(str::to_string)
        .collect()
}

/// The single installed `rustup` toolchain whose name contains `teenyc`. Errors if none or more
/// than one is found — in the latter case the caller needs `TEENYC_PATH` to disambiguate.
fn sole_teenyc_toolchain() -> Result<String> {
    let output = Command::new("rustup")
        .args(["toolchain", "list"])
        .output()
        .context("spawn `rustup toolchain list` (is rustup installed and on PATH?)")?;
    anyhow::ensure!(
        output.status.success(),
        "`rustup toolchain list` exited with {}",
        output.status
    );

    let names = teenyc_toolchain_names(&String::from_utf8_lossy(&output.stdout));
    match names.as_slice() {
        [] => anyhow::bail!(
            "no teenyc rustup toolchain found; set TEENYC_PATH to the teenyc binary, or install \
             one with `cargo teeny install-toolchain` (see cargo-teeny)"
        ),
        [name] => Ok(name.clone()),
        multiple => anyhow::bail!(
            "multiple teenyc rustup toolchains found ({}); set TEENYC_PATH to disambiguate",
            multiple.join(", ")
        ),
    }
}

/// Resolves `toolchain`'s `teenyc` binary path via `rustup which --toolchain <toolchain> teenyc`.
fn which_in_toolchain(toolchain: &str) -> Result<PathBuf> {
    let output = Command::new("rustup")
        .args(["which", "--toolchain", toolchain, "teenyc"])
        .output()
        .with_context(|| format!("spawn `rustup which --toolchain {toolchain} teenyc`"))?;
    anyhow::ensure!(
        output.status.success(),
        "`rustup which --toolchain {toolchain} teenyc` exited with {}",
        output.status
    );
    Ok(PathBuf::from(
        String::from_utf8_lossy(&output.stdout).trim().to_string(),
    ))
}

/// Resolve the effective kernel cache directory.
///
/// Priority: `$TEENYC_CACHE_DIR` (if set) > `<exe_dir>/../cache` (if that
/// directory exists — the layout `cargo teeny package` produces, with
/// `cache/` sitting next to `bin/`) > `/tmp/teenyc_cache`.
///
/// The exe-relative check only ever fires when a real `cache/` directory is
/// actually there, so plain `cargo run`/`cargo test` dev builds (whose exe
/// lives under `target/debug/...`, with no `cache/` sibling) are unaffected.
pub fn default_cache_dir() -> String {
    if let Ok(dir) = std::env::var("TEENYC_CACHE_DIR") {
        return dir;
    }

    match std::env::current_exe() {
        Ok(exe) => sibling_cache_dir(&exe).unwrap_or_else(|| "/tmp/teenyc_cache".to_string()),
        Err(_) => "/tmp/teenyc_cache".to_string(),
    }
}

/// `<exe's parent's parent>/cache`, if that directory exists.
fn sibling_cache_dir(exe: &std::path::Path) -> Option<String> {
    let package_root = exe.parent()?.parent()?;
    let candidate = package_root.join("cache");
    candidate
        .is_dir()
        .then(|| candidate.to_string_lossy().into_owned())
}

#[cfg(test)]
mod find_teenyc_tests {
    use super::*;

    #[test]
    fn finds_single_teenyc_toolchain() {
        let output = "stable-x86_64-unknown-linux-gnu (default)\n\
                       stable-teenyc-x86_64-unknown-linux-gnu\n";
        assert_eq!(
            teenyc_toolchain_names(output),
            vec!["stable-teenyc-x86_64-unknown-linux-gnu"]
        );
    }

    #[test]
    fn empty_when_no_teenyc_toolchain() {
        let output =
            "stable-x86_64-unknown-linux-gnu (default)\nnightly-x86_64-unknown-linux-gnu\n";
        assert!(teenyc_toolchain_names(output).is_empty());
    }

    #[test]
    fn finds_multiple_teenyc_toolchains() {
        let output = "stable-teenyc-x86_64-unknown-linux-gnu (default)\n\
                       my-teenyc-toolchain\n";
        assert_eq!(
            teenyc_toolchain_names(output),
            vec![
                "stable-teenyc-x86_64-unknown-linux-gnu",
                "my-teenyc-toolchain"
            ]
        );
    }
}

#[cfg(test)]
mod cache_dir_tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn tmp_root(name: &str) -> std::path::PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("teeny-compiler-cache-dir-test-{name}-{suffix}"))
    }

    #[test]
    fn finds_sibling_cache_dir_when_present() {
        let root = tmp_root("present");
        let bin_dir = root.join("bin");
        fs::create_dir_all(&bin_dir).unwrap();
        fs::create_dir_all(root.join("cache")).unwrap();
        let exe = bin_dir.join("myapp");

        let found = sibling_cache_dir(&exe).expect("cache dir should be found");
        assert_eq!(found, root.join("cache").to_string_lossy());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn none_when_cache_dir_missing() {
        let root = tmp_root("missing");
        let bin_dir = root.join("bin");
        fs::create_dir_all(&bin_dir).unwrap();
        let exe = bin_dir.join("myapp");

        assert!(sibling_cache_dir(&exe).is_none());

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn none_when_exe_has_no_grandparent() {
        assert!(sibling_cache_dir(std::path::Path::new("myapp")).is_none());
    }
}
