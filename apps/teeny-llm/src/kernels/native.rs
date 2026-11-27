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

//! Kernels built from C in `native/` rather than by the kernel compiler (see
//! `build.rs`), for instructions it cannot express. Each is a shared library with one
//! `*_entry_point`, called like a compiled kernel, embedded in this crate and written
//! to disk on first use so the runtime can load it by path.

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

macro_rules! library {
    ($name:literal) => {
        (
            $name,
            include_bytes!(concat!(env!("OUT_DIR"), "/", $name, ".so")).as_slice(),
        )
    };
}

const LIBRARIES: &[(&str, &[u8])] = &[library!("linear_ime"), library!("quant_pack_i8")];

/// The path of the native kernel library `name`, written under the temp directory on
/// first use, named by its contents' hash so different builds never share a file.
pub fn library(name: &str) -> anyhow::Result<PathBuf> {
    static WRITTEN: OnceLock<Mutex<std::collections::HashMap<String, PathBuf>>> = OnceLock::new();
    let mut written = WRITTEN
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(path) = written.get(name) {
        return Ok(path.clone());
    }
    let bytes = LIBRARIES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, b)| *b)
        .ok_or_else(|| anyhow::anyhow!("no native kernel library {name}"))?;
    anyhow::ensure!(
        !bytes.is_empty(),
        "native kernel library {name} was not built: build with the riscv feature and a \
         RISC-V C compiler ($TEENY_RISCV_CC, default riscv64-linux-gnu-gcc)"
    );
    let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100_0000_01b3)
    });
    let dir = std::env::temp_dir().join("teeny-native");
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{name}-{hash:016x}.so"));
    if !path.exists() {
        // Written aside and renamed, so a concurrent process never loads half a file.
        let tmp = dir.join(format!("{name}-{hash:016x}.{}.tmp", std::process::id()));
        std::fs::write(&tmp, bytes)?;
        std::fs::rename(&tmp, &path)?;
    }
    written.insert(name.to_string(), path.clone());
    Ok(path)
}
