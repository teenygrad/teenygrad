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

//! Builds the native RISC-V kernels in `native/` (those the kernel compiler cannot
//! express, such as the SpacemiT A100 matrix-engine linear) into shared libraries the
//! crate embeds; see `kernels::native`.
//!
//! Only with the `riscv` or `qemu` feature, with `$TEENY_RISCV_CC` (default
//! `riscv64-linux-gnu-gcc`). Without a compiler each library is left empty and the
//! ops that need one fail to load, saying so.

use std::path::PathBuf;
use std::process::Command;

const KERNELS: &[&str] = &["linear_ime", "quant_pack_i8"];

fn main() {
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    let wanted = std::env::var_os("CARGO_FEATURE_RISCV").is_some()
        || std::env::var_os("CARGO_FEATURE_QEMU").is_some();
    let cc = std::env::var("TEENY_RISCV_CC").unwrap_or_else(|_| "riscv64-linux-gnu-gcc".to_string());
    println!("cargo::rerun-if-env-changed=TEENY_RISCV_CC");
    for name in KERNELS {
        let src = format!("native/{name}.c");
        println!("cargo::rerun-if-changed={src}");
        let lib = out.join(format!("{name}.so"));
        let built = wanted
            && Command::new(&cc)
                .args(["-O3", "-march=rv64gcv", "-mabi=lp64d", "-shared", "-fPIC", "-nostartfiles", "-o"])
                .arg(&lib)
                .arg(&src)
                .arg("-lm")
                .status()
                .is_ok_and(|s| s.success());
        if !built {
            if wanted {
                println!("cargo::warning=could not build native kernel {name} with {cc}; ops that use it will fail to load");
            }
            std::fs::write(&lib, []).expect("writing an empty placeholder library");
        }
    }
}
