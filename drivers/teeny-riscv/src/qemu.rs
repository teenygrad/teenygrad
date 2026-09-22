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

//! Runs compiled RISC-V kernels under `qemu-riscv64` user-mode emulation.
//!
//! A kernel's shared library is a RISC-V ELF, so it can't be loaded into this process. Instead
//! [`launch`] cross-compiles a small C trampoline for the kernel's argument types (once per
//! signature, cached next to the compiled kernels), writes the buffers the kernel's pointer
//! arguments address to files, and runs the trampoline under `qemu-riscv64`. The trampoline
//! calls the entry point once per program id and writes the buffers back, and [`launch`] copies
//! them into the original buffers.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result, anyhow, bail, ensure};
use teeny_core::device::program::{ArgVisitor, KernelArgs};

use crate::device::buffer::{BufferRegistry, Region};
use crate::errors::Error;

/// Runs the kernel `entry_point` from `library` once for every program id in `grid`, with `args`.
///
/// Every pointer argument must point into a buffer registered in `buffers`; whatever the kernel
/// writes to those buffers is copied back before this returns.
pub(crate) fn launch<A: KernelArgs>(
    buffers: &BufferRegistry,
    library: &Path,
    entry_point: &str,
    grid: [u32; 3],
    args: &A,
) -> Result<()> {
    let mut collector = ArgCollector::default();
    args.visit_args(&mut collector);
    let args = collector.0;

    let env = QemuEnv::get()?;
    let trampoline = env.trampoline(&args)?;

    // Holding the registry lock for the whole run keeps every buffer involved alive: a buffer
    // unregisters itself under this lock before its memory is freed.
    let registry = buffers.lock();
    let mut regions: Vec<Region> = Vec::new();
    let mut values = Vec::with_capacity(args.len());
    for (index, arg) in args.iter().enumerate() {
        values.push(match *arg {
            Arg::Ptr(addr) => {
                let region = BufferRegistry::find_in(&registry, addr)
                    .ok_or(Error::UnknownBufferPointer { index, addr })?;
                let slot = regions
                    .iter()
                    .position(|r| *r == region)
                    .unwrap_or_else(|| {
                        regions.push(region);
                        regions.len() - 1
                    });
                format!("{slot}:{}", addr - region.addr)
            }
            Arg::Bool(v) => u8::from(v).to_string(),
            Arg::I8(v) => v.to_string(),
            Arg::I16(v) => v.to_string(),
            Arg::I32(v) => v.to_string(),
            Arg::I64(v) => v.to_string(),
            Arg::U8(v) => v.to_string(),
            Arg::U16(v) => v.to_string(),
            Arg::U32(v) => v.to_string(),
            Arg::U64(v) => v.to_string(),
            Arg::F32(v) => format!("{:x}", v.to_bits()),
            Arg::F64(v) => format!("{:x}", v.to_bits()),
        });
    }

    let files = env.buffer_files(regions.len());
    let result = run(
        env,
        &trampoline,
        library,
        entry_point,
        grid,
        &regions,
        &files,
        &values,
    );
    for file in &files {
        let _ = std::fs::remove_file(file);
    }
    drop(registry);
    result
}

#[allow(clippy::too_many_arguments)]
fn run(
    env: &QemuEnv,
    trampoline: &Path,
    library: &Path,
    entry_point: &str,
    grid: [u32; 3],
    regions: &[Region],
    files: &[PathBuf],
    values: &[String],
) -> Result<()> {
    for (region, file) in regions.iter().zip(files) {
        // Safety: `region` is a live buffer `region.bytes` long, kept alive by the registry lock
        // `launch` holds.
        let bytes = unsafe { std::slice::from_raw_parts(region.addr as *const u8, region.bytes) };
        std::fs::write(file, bytes)
            .with_context(|| format!("failed to write {}", file.display()))?;
    }

    let output = Command::new(&env.qemu)
        // The kernels use RVV instructions; `max` enables every extension QEMU emulates.
        .args(["-cpu", "max"])
        .arg("-L")
        .arg(&env.sysroot)
        .arg(trampoline)
        .arg(library)
        .arg(entry_point)
        .args(grid.map(|n| n.to_string()))
        .arg(regions.len().to_string())
        .args(files)
        .args(values)
        .output()
        .with_context(|| format!("failed to run {} under {}", trampoline.display(), env.qemu))?;
    if !output.status.success() {
        bail!(
            "{entry_point} failed under qemu-riscv64 ({}): {}{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    for (region, file) in regions.iter().zip(files) {
        let bytes =
            std::fs::read(file).with_context(|| format!("failed to read {}", file.display()))?;
        ensure!(
            bytes.len() == region.bytes,
            "{} holds {} bytes after the kernel ran, expected {}",
            file.display(),
            bytes.len(),
            region.bytes
        );
        // Safety: `region` is a live buffer `region.bytes` long (see above) whose elements are
        // `UnsafeCell`s, so it may be written through its address while the buffer is shared.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), region.addr as *mut u8, region.bytes)
        };
    }
    Ok(())
}

/// One kernel argument, as reported by [`KernelArgs::visit_args`].
#[derive(Debug, Clone, Copy)]
enum Arg {
    Ptr(usize),
    Bool(bool),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
}

impl Arg {
    /// The C type the trampoline declares for this argument, so the entry point receives it
    /// exactly as the RISC-V calling convention expects.
    fn c_type(&self) -> &'static str {
        match self {
            Arg::Ptr(_) => "void *",
            Arg::Bool(_) => "_Bool",
            Arg::I8(_) => "int8_t",
            Arg::I16(_) => "int16_t",
            Arg::I32(_) => "int32_t",
            Arg::I64(_) => "int64_t",
            Arg::U8(_) => "uint8_t",
            Arg::U16(_) => "uint16_t",
            Arg::U32(_) => "uint32_t",
            Arg::U64(_) => "uint64_t",
            Arg::F32(_) => "float",
            Arg::F64(_) => "double",
        }
    }
}

#[derive(Default)]
struct ArgCollector(Vec<Arg>);

impl ArgVisitor for ArgCollector {
    fn visit_ptr(&mut self, ptr: *mut core::ffi::c_void) {
        self.0.push(Arg::Ptr(ptr as usize));
    }
    fn visit_bool(&mut self, val: bool) {
        self.0.push(Arg::Bool(val));
    }
    fn visit_i8(&mut self, val: i8) {
        self.0.push(Arg::I8(val));
    }
    fn visit_i16(&mut self, val: i16) {
        self.0.push(Arg::I16(val));
    }
    fn visit_i32(&mut self, val: i32) {
        self.0.push(Arg::I32(val));
    }
    fn visit_i64(&mut self, val: i64) {
        self.0.push(Arg::I64(val));
    }
    fn visit_u8(&mut self, val: u8) {
        self.0.push(Arg::U8(val));
    }
    fn visit_u16(&mut self, val: u16) {
        self.0.push(Arg::U16(val));
    }
    fn visit_u32(&mut self, val: u32) {
        self.0.push(Arg::U32(val));
    }
    fn visit_u64(&mut self, val: u64) {
        self.0.push(Arg::U64(val));
    }
    fn visit_f32(&mut self, val: f32) {
        self.0.push(Arg::F32(val));
    }
    fn visit_f64(&mut self, val: f64) {
        self.0.push(Arg::F64(val));
    }
}

/// The host tools a launch needs, resolved once per process.
struct QemuEnv {
    /// Cross C compiler for `riscv64-linux-gnu`, which builds the trampolines.
    cc: String,
    /// `qemu-riscv64`.
    qemu: String,
    /// The cross toolchain's sysroot, where `qemu-riscv64 -L` finds the RISC-V dynamic linker and
    /// libc the trampoline needs to `dlopen` the kernel.
    sysroot: PathBuf,
    /// Where trampolines and launch buffers are written.
    cache_dir: PathBuf,
}

impl QemuEnv {
    fn get() -> Result<&'static QemuEnv> {
        static ENV: OnceLock<std::result::Result<QemuEnv, String>> = OnceLock::new();
        ENV.get_or_init(|| QemuEnv::resolve().map_err(|e| format!("{e:#}")))
            .as_ref()
            .map_err(|e| anyhow!("cannot run RISC-V kernels under QEMU: {e}"))
    }

    /// Both tools default to their usual names on `PATH` and can be overridden with
    /// `TEENYC_RISCV_CC` and `TEENYC_QEMU_RISCV64`, the same variables `teeny-test` reads.
    fn resolve() -> Result<QemuEnv> {
        let cc = resolve_tool("TEENYC_RISCV_CC", "riscv64-linux-gnu-gcc")?;
        let qemu = resolve_tool("TEENYC_QEMU_RISCV64", "qemu-riscv64")?;
        let sysroot = cc_sysroot(&cc)?;
        let cache_dir = PathBuf::from(teeny_compiler::compiler::default_cache_dir());
        std::fs::create_dir_all(&cache_dir)
            .with_context(|| format!("failed to create cache dir {}", cache_dir.display()))?;
        Ok(QemuEnv {
            cc,
            qemu,
            sysroot,
            cache_dir,
        })
    }

    /// The trampoline binary for kernels taking `args`, compiled on first use.
    fn trampoline(&self, args: &[Arg]) -> Result<PathBuf> {
        let source = trampoline_source(args);
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        let name = format!("riscv_qemu_trampoline_{:016x}", hasher.finish());
        let binary = self.cache_dir.join(&name);
        if binary.is_file() {
            return Ok(binary);
        }

        // Build under per-process names and rename into place, so a concurrent launch never runs
        // a partially written binary.
        let pid = std::process::id();
        let source_file = self.cache_dir.join(format!("{name}.{pid}.c"));
        let tmp_binary = self.cache_dir.join(format!("{name}.{pid}.tmp"));
        std::fs::write(&source_file, &source)
            .with_context(|| format!("failed to write {}", source_file.display()))?;
        let output = Command::new(&self.cc)
            .args(["-O2", "-o"])
            .arg(&tmp_binary)
            .arg(&source_file)
            .arg("-ldl")
            .output()
            .with_context(|| format!("failed to run {}", self.cc))?;
        let _ = std::fs::remove_file(&source_file);
        if !output.status.success() {
            bail!(
                "{} failed to compile the QEMU trampoline: {}",
                self.cc,
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::fs::rename(&tmp_binary, &binary)
            .with_context(|| format!("failed to move the trampoline to {}", binary.display()))?;
        Ok(binary)
    }

    /// `count` scratch file paths unique to one launch in this process.
    fn buffer_files(&self, count: usize) -> Vec<PathBuf> {
        static NEXT_LAUNCH: AtomicUsize = AtomicUsize::new(0);
        let launch = NEXT_LAUNCH.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        (0..count)
            .map(|i| {
                self.cache_dir
                    .join(format!("riscv_qemu_launch_{pid}_{launch}_buffer{i}.bin"))
            })
            .collect()
    }
}

fn resolve_tool(env_var: &str, default_name: &str) -> Result<String> {
    if let Ok(path) = std::env::var(env_var) {
        ensure!(
            Path::new(&path).is_file(),
            "{env_var}={path:?} does not point to an existing file"
        );
        return Ok(path);
    }
    std::env::var_os("PATH")
        .iter()
        .flat_map(std::env::split_paths)
        .map(|dir| dir.join(default_name))
        .find(|candidate| candidate.is_file())
        .map(|path| path.to_string_lossy().into_owned())
        .ok_or_else(|| {
            anyhow!("`{default_name}` was not found on PATH; install it or set {env_var}")
        })
}

/// Debian/Ubuntu-packaged cross toolchains (`gcc-riscv64-linux-gnu`) install their target libc
/// under `/usr/<multiarch-triplet>` and `-print-sysroot` just prints `/`, so `-print-multiarch` is
/// tried first, then `-print-sysroot` for toolchains that do support it.
fn cc_sysroot(cc: &str) -> Result<PathBuf> {
    let multiarch = run_cc(cc, "-print-multiarch")?;
    if !multiarch.is_empty() {
        let sysroot = PathBuf::from("/usr").join(&multiarch);
        if sysroot.is_dir() {
            return Ok(sysroot);
        }
    }

    let sysroot = run_cc(cc, "-print-sysroot")?;
    if !sysroot.is_empty() && sysroot != "/" && Path::new(&sysroot).is_dir() {
        return Ok(PathBuf::from(sysroot));
    }

    bail!(
        "could not determine a RISC-V sysroot for `{cc}` (checked `/usr/{multiarch}` and \
         `-print-sysroot`); install the matching cross libc (e.g. `libc6-dev-riscv64-cross`)"
    );
}

fn run_cc(cc: &str, arg: &str) -> Result<String> {
    let output = Command::new(cc)
        .arg(arg)
        .output()
        .with_context(|| format!("failed to run `{cc} {arg}`"))?;
    ensure!(
        output.status.success(),
        "`{cc} {arg}` exited with {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// The C source of the trampoline for kernels taking `args`: see [`TRAMPOLINE_TEMPLATE`].
fn trampoline_source(args: &[Arg]) -> String {
    let mut params = String::new();
    let mut parse = String::new();
    let mut call = String::new();
    for (i, arg) in args.iter().enumerate() {
        let c_type = arg.c_type();
        let value = match arg {
            Arg::Ptr(_) => format!("pointer_arg(argv[first_arg + {i}], buffers, n_buffers)"),
            Arg::F32(_) => format!("float_arg(argv[first_arg + {i}])"),
            Arg::F64(_) => format!("double_arg(argv[first_arg + {i}])"),
            Arg::U64(_) => format!("(uint64_t)strtoull(argv[first_arg + {i}], NULL, 10)"),
            _ => format!("({c_type})strtoll(argv[first_arg + {i}], NULL, 10)"),
        };
        params.push_str(&format!("{c_type}, "));
        parse.push_str(&format!("    {c_type} a{i} = {value};\n"));
        call.push_str(&format!("a{i}, "));
    }
    TRAMPOLINE_TEMPLATE
        .replace("@PARAMS@", &params)
        .replace("@N_ARGS@", &args.len().to_string())
        .replace("@PARSE@", &parse)
        .replace("@CALL@", &call)
}

/// Loads the buffer files, calls the entry point once per program id, and writes the buffers back.
///
/// Arguments: `<library> <entry-point> <grid-x> <grid-y> <grid-z> <n-buffers> <buffer files...>
/// <kernel arguments...>`. A pointer argument is `<buffer>:<byte offset>`, an integer is decimal,
/// and a float is the hex of its bits so every value round-trips exactly.
const TRAMPOLINE_TEMPLATE: &str = r#"
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The entry point's own arguments, then the program id and grid size (x, y, z) that TritonCPU
   appends when it lowers a kernel. */
typedef void (*kernel_fn)(@PARAMS@int32_t, int32_t, int32_t, uint32_t, uint32_t, uint32_t);

struct buffer {
    const char *path;
    void *data;
    size_t size;
};

__attribute__((unused)) static int load_buffer(struct buffer *b) {
    FILE *f = fopen(b->path, "rb");
    if (!f) {
        perror(b->path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    rewind(f);
    if (size < 0 || posix_memalign(&b->data, 64, size > 0 ? (size_t)size : 1) != 0) {
        fprintf(stderr, "%s: cannot allocate %ld bytes\n", b->path, size);
        fclose(f);
        return -1;
    }
    b->size = (size_t)size;
    if (fread(b->data, 1, b->size, f) != b->size) {
        fprintf(stderr, "%s: short read\n", b->path);
        fclose(f);
        return -1;
    }
    fclose(f);
    return 0;
}

__attribute__((unused)) static int save_buffer(const struct buffer *b) {
    FILE *f = fopen(b->path, "wb");
    if (!f || fwrite(b->data, 1, b->size, f) != b->size || fclose(f) != 0) {
        fprintf(stderr, "%s: cannot write the kernel's output\n", b->path);
        return -1;
    }
    return 0;
}

__attribute__((unused)) static void *pointer_arg(const char *spec, struct buffer *buffers, long n_buffers) {
    char *end;
    long slot = strtol(spec, &end, 10);
    if (*end != ':' || slot < 0 || slot >= n_buffers) {
        fprintf(stderr, "bad pointer argument \"%s\"\n", spec);
        exit(2);
    }
    unsigned long long offset = strtoull(end + 1, NULL, 10);
    if (offset > buffers[slot].size) {
        fprintf(stderr, "pointer argument \"%s\" is past the end of its buffer\n", spec);
        exit(2);
    }
    return (char *)buffers[slot].data + offset;
}

__attribute__((unused)) static float float_arg(const char *bits) {
    uint32_t raw = (uint32_t)strtoul(bits, NULL, 16);
    float value;
    memcpy(&value, &raw, sizeof value);
    return value;
}

__attribute__((unused)) static double double_arg(const char *bits) {
    uint64_t raw = (uint64_t)strtoull(bits, NULL, 16);
    double value;
    memcpy(&value, &raw, sizeof value);
    return value;
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr,
                "usage: %s <library> <entry-point> <grid-x> <grid-y> <grid-z> <n-buffers> "
                "<buffer files...> <kernel arguments...>\n",
                argv[0]);
        return 2;
    }
    long n_buffers = strtol(argv[6], NULL, 10);
    int first_arg = 7 + (int)n_buffers;
    if (n_buffers < 0 || argc != first_arg + @N_ARGS@) {
        fprintf(stderr, "expected %ld buffer files and @N_ARGS@ kernel arguments\n", n_buffers);
        return 2;
    }

    /* Kernels call libm (e.g. `expf`) without listing it as a dependency, so make its symbols
       globally available before loading the kernel. */
    if (!dlopen("libm.so.6", RTLD_NOW | RTLD_GLOBAL)) {
        fprintf(stderr, "dlopen(libm.so.6) failed: %s\n", dlerror());
        return 1;
    }
    void *handle = dlopen(argv[1], RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "dlopen(%s) failed: %s\n", argv[1], dlerror());
        return 1;
    }
    kernel_fn kernel = (kernel_fn)dlsym(handle, argv[2]);
    if (!kernel) {
        fprintf(stderr, "dlsym(%s) failed: %s\n", argv[2], dlerror());
        return 1;
    }
    uint32_t grid[3] = {
        (uint32_t)strtoul(argv[3], NULL, 10),
        (uint32_t)strtoul(argv[4], NULL, 10),
        (uint32_t)strtoul(argv[5], NULL, 10),
    };

    struct buffer *buffers = calloc(n_buffers > 0 ? (size_t)n_buffers : 1, sizeof *buffers);
    if (!buffers) {
        fprintf(stderr, "cannot allocate %ld buffers\n", n_buffers);
        return 1;
    }
    for (long i = 0; i < n_buffers; ++i) {
        buffers[i].path = argv[7 + i];
        if (load_buffer(&buffers[i]) != 0) {
            return 1;
        }
    }

@PARSE@
    for (uint32_t z = 0; z < grid[2]; ++z) {
        for (uint32_t y = 0; y < grid[1]; ++y) {
            for (uint32_t x = 0; x < grid[0]; ++x) {
                kernel(@CALL@(int32_t)x, (int32_t)y, (int32_t)z, grid[0], grid[1], grid[2]);
            }
        }
    }

    for (long i = 0; i < n_buffers; ++i) {
        if (save_buffer(&buffers[i]) != 0) {
            return 1;
        }
    }
    return 0;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trampoline_declares_the_kernel_arguments_then_the_program_context() {
        let source = trampoline_source(&[Arg::Ptr(0), Arg::I32(4), Arg::F32(0.5)]);
        assert!(source.contains(
            "typedef void (*kernel_fn)(void *, int32_t, float, int32_t, int32_t, int32_t, \
             uint32_t, uint32_t, uint32_t);"
        ));
        assert!(source.contains("argc != first_arg + 3"));
        assert!(source.contains("kernel(a0, a1, a2, (int32_t)x,"));
    }
}
