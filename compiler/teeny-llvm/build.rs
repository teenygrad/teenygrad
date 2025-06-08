/*
 * Copyright (C) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use cargo_metadata::MetadataCommand;
use cfgrammar::yacc::YaccKind;
use lrlex::CTLexerBuilder;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR environment variable not set"));
    let metadata = MetadataCommand::new().exec().unwrap();

    let package = metadata
        .packages
        .iter()
        .filter(|p| p.name.as_str() == "teeny-llvm")
        .next()
        .unwrap();
    let project_dir: PathBuf = package.manifest_path.parent().unwrap().into();

    // Get workspace root from metadata
    let workspace_root: PathBuf = metadata.workspace_root.into();
    let build_dir: PathBuf = workspace_root.join("target/cmake-build");
    std::fs::create_dir_all(&build_dir).expect("Failed to create build directory");

    println!("AXM project_dir: {:?}", project_dir);
    println!("AXM build_dir: {:?}", build_dir);

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");
    // Tell cargo to invalidate the built crate whenever toy language files change
    println!("cargo:rerun-if-changed=examples/toy/toy.l");
    println!("cargo:rerun-if-changed=examples/toy/toy.y");

    // Check if cmake and ninja are installed
    check_command("cmake");
    check_command("ninja");

    build_llvm(&project_dir, &build_dir);
    build_triton(&project_dir, &build_dir);
    build_tutorials(&project_dir, &build_dir);
    build_toy_lang(&out_dir);

    generate_bindings(&build_dir);
}

fn check_command(command: &str) {
    let output = Command::new(command)
        .arg("--version")
        .output()
        .unwrap_or_else(|e| panic!("Failed to execute {} --version command: {}", command, e));

    if !output.status.success() {
        panic!(
            "{} is not installed or not working properly. Error: {}",
            command,
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn build_llvm(project_dir: &Path, build_dir: &Path) {
    let modules_dir = project_dir.join("modules");

    let status = Command::new("./scripts/build_llvm.sh")
        .env("BUILD_DIR", build_dir)
        .env("MODULES_DIR", modules_dir)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .expect("Failed to execute cmake command")
        .wait()
        .unwrap_or_else(|e| panic!("Failed to wait for cmake command: {}", e));

    if !status.success() {
        panic!(
            "LLVM build failed with exit code: {}",
            status.code().unwrap_or(-1)
        );
    }
}

fn build_triton(project_dir: &Path, build_dir: &Path) {
    let modules_dir = project_dir.join("modules");

    let status = Command::new("./scripts/build_triton.sh")
        .env("BUILD_DIR", build_dir)
        .env("MODULES_DIR", modules_dir)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .expect("Triton build: Failed to execute cmake command")
        .wait()
        .unwrap_or_else(|e| panic!("Triton build: Failed to wait for cmake command: {}", e));

    if !status.success() {
        panic!(
            "Triton build failed with exit code: {}",
            status.code().unwrap_or(-1)
        );
    }
}

fn build_tutorials(project_dir: &Path, build_dir: &Path) {
    let modules_dir = project_dir.join("modules");
    let tutorials_dir = project_dir.join("tutorials");

    let status = Command::new("./scripts/build_tutorials.sh")
        .env("BUILD_DIR", build_dir)
        .env("MODULES_DIR", modules_dir)
        .env("TUTORIALS_DIR", tutorials_dir)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .expect("Failed to execute cmake command")
        .wait()
        .unwrap_or_else(|e| panic!("Failed to wait for cmake command: {}", e));

    if !status.success() {
        panic!(
            "Tutorials build failed with exit code: {}",
            status.code().unwrap_or(-1)
        );
    }
}

fn build_toy_lang(out_dir: &Path) {
    let out_path1 = out_dir.join("toy_y.rs");
    let out_path2 = out_dir.join("toy_l.rs");

    let _lexer = CTLexerBuilder::new()
        .lrpar_config(move |ctp| {
            ctp.grammar_path("tests/toy/toy.y")
                .yacckind(YaccKind::Grmtools)
                .output_path(out_path1.clone())
        })
        .lexer_path("tests/toy/toy.l")
        .output_path(out_path2)
        .build()
        .unwrap_or_else(|e| panic!("Failed to build lexer: {}", e));
}

fn generate_bindings(build_dir: &Path) {
    let llvm_include_dir = String::from(build_dir.join("install/include").to_str().unwrap());
    let llvm_lib_dir = String::from(build_dir.join("install/lib").to_str().unwrap());

    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search={llvm_lib_dir}");
    println!("cargo:rustc-link-search={llvm_lib_dir}/stubs");

    // Tell cargo to tell rustc to link the cuda and nvrtc libraries
    println!("cargo:rustc-link-lib=LLVMCore");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        .clang_arg(format!("-I{llvm_include_dir}"))
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.h")
        .derive_default(true)
        // .derive_debug(true)
        // .derive_eq(true)
        // .derive_partialeq(true)
        // .derive_hash(true)
        // .derive_partialhash(true)
        // .derive_clone(true)
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    bindings
        .write_to_file(build_dir.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
