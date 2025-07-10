extern crate bindgen;

use std::env;
use std::path::PathBuf;

fn main() {
    let cuda_include_dir =
        env::var("CUDA_INCLUDE_DIR").unwrap_or_else(|_| "/usr/local/cuda/include".to_string());
    let cuda_lib_dir =
        env::var("CUDA_LIB_DIR").unwrap_or_else(|_| "/usr/local/cuda/lib64".to_string());

    // Tell cargo to look for shared libraries in the specified directory
    println!("cargo:rustc-link-search={cuda_lib_dir}");
    println!("cargo:rustc-link-search={cuda_lib_dir}/stubs");

    // Tell cargo to tell rustc to link the cuda and nvrtc libraries
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=nvrtc");

    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed=wrapper.h");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        .clang_arg(format!("-I{cuda_include_dir}"))
        // The input header we would like to generate
        // bindings for.
        .header("wrapper.h")
        // derive default trait whenever possible
        .derive_default(true)
        // filter functions which make use of u128 due to FFI ABI issues
        .blocklist_function("strtold")
        .blocklist_function("qecvt")
        .blocklist_function("qfcvt")
        .blocklist_function("qgcvt")
        .blocklist_function("qecvt_r")
        .blocklist_function("qfcvt_r")
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    let bindings_dir = out_path.display();
    println!("Bindings written to {bindings_dir}")
}
