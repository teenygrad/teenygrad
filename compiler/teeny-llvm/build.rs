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

use std::path::{Path, PathBuf};
use std::process::Command;

use cargo_metadata::MetadataCommand;

fn main() {
    let metadata = MetadataCommand::new().exec().unwrap();

    let package = metadata
        .packages
        .iter()
        .find(|p| p.name.as_str() == "teeny-llvm")
        .unwrap();
    let project_dir: PathBuf = package.manifest_path.parent().unwrap().into();

    // Get workspace root from metadata
    let workspace_root: PathBuf = metadata.workspace_root.into();
    let build_dir: PathBuf = workspace_root.join("target/cmake-build");
    std::fs::create_dir_all(&build_dir).expect("Failed to create build directory");

    let out_dir: PathBuf = std::env::var("OUT_DIR").unwrap().into();

    println!("AXM project_dir: {:?}", project_dir);
    println!("AXM build_dir: {:?}", build_dir);

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");

    // Check if cmake and ninja are installed
    check_command("cmake");
    check_command("ninja");

    if !check_build_done(&build_dir, "llvm") {
        build_llvm(&project_dir, &build_dir);
    }

    if !check_build_done(&build_dir, "triton") {
        build_triton(&project_dir, &build_dir);
    }

    if !check_build_done(&build_dir, "teeny") {
        build_teeny(&project_dir, &build_dir);
    }

    if !check_build_done(&build_dir, "tutorials") {
        build_tutorials(&project_dir, &build_dir);
    }

    generate_bindings(&build_dir, &out_dir);
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

fn build_teeny(project_dir: &Path, build_dir: &Path) {
    let modules_dir = project_dir.join("modules");

    let status = Command::new("./scripts/build_teeny.sh")
        .env("BUILD_DIR", build_dir)
        .env("MODULES_DIR", modules_dir)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .expect("Teeny build: Failed to execute cmake command")
        .wait()
        .unwrap_or_else(|e| panic!("Teeny build: Failed to wait for cmake command: {}", e));

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

fn generate_bindings(build_dir: &Path, out_dir: &Path) {
    let llvm_include_dir = String::from(build_dir.join("install/include").to_str().unwrap());
    let llvm_lib_dir = String::from(build_dir.join("install/lib").to_str().unwrap());

    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search={llvm_lib_dir}");
    println!("cargo:rustc-link-search={llvm_lib_dir}/stubs");

    // Tell cargo to tell rustc to link the cuda and nvrtc libraries
    println!("cargo:rustc-link-lib=teeny");
    println!("cargo:rustc-link-lib=triton");

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
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn check_build_done(build_dir: &Path, project: &str) -> bool {
    build_dir.join(format!("{}/build.done", project)).exists()
}
