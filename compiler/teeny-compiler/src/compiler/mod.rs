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

pub mod backend;
pub mod driver;
pub mod target;

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
