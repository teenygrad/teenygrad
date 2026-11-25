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

//! Builds the Vue app in `ui/` with npm and embeds it: writes `$OUT_DIR/assets.rs`, the
//! table of every file of the build, which `src/lib.rs` includes.
//!
//! `TEENY_WEBAPP_UI=skip` (or building on docs.rs) embeds a placeholder page instead, for
//! builds without Node that don't need the UI (API docs).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs, io};

/// The UI's inputs: a change to any rebuilds it.
const INPUTS: [&str; 9] = [
    "src",
    "public",
    "index.html",
    "env.d.ts",
    "package.json",
    "package-lock.json",
    "vite.config.ts",
    "tsconfig.app.json",
    "tsconfig.json",
];

const PLACEHOLDER: &str = "<!DOCTYPE html>\n<html lang=\"en\">\n<head><meta charset=\"UTF-8\"><title>teenygrad</title></head>\n<body><p>This teeny-webapp was built without its UI (<code>TEENY_WEBAPP_UI=skip</code>). Rebuild it with Node installed to embed the app.</p></body>\n</html>\n";

fn main() {
    let manifest =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    let ui = manifest.join("ui");
    for input in INPUTS {
        println!("cargo::rerun-if-changed=ui/{input}");
    }
    println!("cargo::rerun-if-env-changed=TEENY_WEBAPP_UI");
    println!("cargo::rerun-if-env-changed=DOCS_RS");

    let dist = out.join("dist");
    let skip =
        env::var("TEENY_WEBAPP_UI").is_ok_and(|v| v == "skip") || env::var_os("DOCS_RS").is_some();
    if skip {
        fs::create_dir_all(&dist).expect("creating the placeholder's folder");
        fs::write(dist.join("index.html"), PLACEHOLDER).expect("writing the placeholder page");
    } else {
        build(&ui, &dist);
    }

    let mut files = Vec::new();
    walk(&dist, &dist, &mut files).expect("listing the UI build");
    files.sort();
    let mut table = String::from(
        "/// Every file of the UI build, by path.\npub static ASSETS: &[Asset] = &[\n",
    );
    for (path, file) in &files {
        writeln!(
            table,
            "    Asset {{ path: {path:?}, mime: {:?}, bytes: include_bytes!({:?}) }},",
            mime(path),
            file.display().to_string()
        )
        .expect("writing to a String");
    }
    table.push_str("];\n");
    fs::write(out.join("assets.rs"), table).expect("writing assets.rs");
}

/// Runs `npm` in `ui` with `args`, failing the build with what went wrong.
fn npm(ui: &Path, args: &[&str]) {
    let program = if cfg!(windows) { "npm.cmd" } else { "npm" };
    match Command::new(program).args(args).current_dir(ui).status() {
        Ok(status) if status.success() => {}
        Ok(status) => panic!(
            "`npm {}` in {} failed ({status})",
            args.join(" "),
            ui.display()
        ),
        Err(e) if e.kind() == io::ErrorKind::NotFound => panic!(
            "teeny-webapp builds its UI with npm, which isn't installed: install Node \
             (^20.19 or >=22.12), or set TEENY_WEBAPP_UI=skip to embed a placeholder page"
        ),
        Err(e) => panic!("running `npm {}`: {e}", args.join(" ")),
    }
}

/// Whether `ui`'s dependencies need (re)installing: never installed, or installed before
/// the lock file last changed.
fn stale_dependencies(ui: &Path) -> bool {
    let modified = |p: PathBuf| fs::metadata(p).and_then(|m| m.modified()).ok();
    match (
        modified(ui.join("node_modules/.package-lock.json")),
        modified(ui.join("package-lock.json")),
    ) {
        (Some(installed), Some(locked)) => installed < locked,
        _ => true,
    }
}

/// Builds the app in `ui` into `dist`.
fn build(ui: &Path, dist: &Path) {
    if stale_dependencies(ui) {
        npm(ui, &["ci", "--no-audit", "--no-fund"]);
    }
    let out_dir = dist.display().to_string();
    npm(
        ui,
        &[
            "run",
            "build-only",
            "--",
            "--outDir",
            &out_dir,
            "--emptyOutDir",
        ],
    );
}

/// Every file under `dir`, as its path relative to `root` (with `/`) and on disk.
fn walk(root: &Path, dir: &Path, files: &mut Vec<(String, PathBuf)>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            walk(root, &path, files)?;
        } else {
            let relative = path
                .strip_prefix(root)
                .expect("walked from root")
                .components()
                .map(|c| c.as_os_str().to_string_lossy())
                .collect::<Vec<_>>()
                .join("/");
            files.push((relative, path));
        }
    }
    Ok(())
}

/// The media type of a file, by its extension.
fn mime(path: &str) -> &'static str {
    match path.rsplit_once('.').map(|(_, ext)| ext) {
        Some("html") => "text/html; charset=utf-8",
        Some("js" | "mjs") => "text/javascript; charset=utf-8",
        Some("css") => "text/css; charset=utf-8",
        Some("json" | "map") => "application/json",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        Some("ico") => "image/x-icon",
        Some("woff2") => "font/woff2",
        Some("woff") => "font/woff",
        Some("txt") => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    }
}
