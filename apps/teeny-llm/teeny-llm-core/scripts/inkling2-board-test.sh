#!/usr/bin/env bash
# Runs the Inkling-2 kernel and graph tests on a RISC-V board (e.g. a SpacemiT K3).
#
# qemu has already been shown to mislead about this code: the RVV NVFP4 kernels measured
# 0.53-0.65x of scalar under emulation and 1.48x on an X100, the wrong side of 1. It models
# no cache and runs vector instructions in software, so only a board settles anything.
#
# Like functiongemma-board-serve.sh, the board needs nothing installed beyond glibc: this
# cross-compiles the test binaries for riscv64gc-unknown-linux-gnu, compiles every kernel
# here (where teenyc lives) by running those binaries under qemu-user, then stages the
# binaries, the kernel cache and the test fixtures. On the board it runs with teenyc
# disabled, so a cache miss fails loudly instead of trying to compile.
#
# Usage: apps/teeny-llm/teeny-llm-core/scripts/inkling2-board-test.sh [-H SSH_HOST]
#   Without -H, only builds and stages (in target/riscv64-native/inkling2-test-stage).
#   Needs: rustup target riscv64gc-unknown-linux-gnu, gcc-riscv64-linux-gnu,
#          qemu-user(-static); teenyc (TEENYC_PATH, or found by teeny-compiler).
#   Env:   TEENYC_CAPABILITY (default spacemit-k3); REMOTE_DIR (default inkling2-tests).
set -euo pipefail

host=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -H) host="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

cd "$(dirname "$0")/../../../.."
triple=riscv64gc-unknown-linux-gnu
capability=${TEENYC_CAPABILITY:-spacemit-k3}
remote=${REMOTE_DIR:-inkling2-tests}
target_dir=$PWD/target/riscv64-native
stage=$target_dir/inkling2-test-stage
cache=$target_dir/kernel-cache
fixtures=$PWD/apps/teeny-llm/teeny-llm-core/tests/fixtures
qemu=$(command -v qemu-riscv64-static || command -v qemu-riscv64)
# The board has VLEN=256 (vlenb reads 32) and the kernels are compiled for it with
# `zvl256b`, so qemu must be given the same width. `-cpu max` defaults to 128 and the
# code compiled for 256 then misbehaves -- as 33 of 47 kernel tests failing, not as a
# clean error.
qemu_cpu=${QEMU_CPU:-rv64,v=true,vlen=256,elen=64,zba=true,zbb=true,zbs=true}
sysroot=/usr/riscv64-linux-gnu
tests=(test_inkling2_kernels test_inkling2_reference test_inkling2_model)

# ── Cross-compile the test binaries ───────────────────────────────────────────────────────────
for t in "${tests[@]}"; do
    CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER=riscv64-linux-gnu-gcc \
        CARGO_TARGET_DIR=$target_dir \
        cargo test --release --no-run --target $triple -p teeny-llm-core \
        --no-default-features --features inkling2,riscv --test "$t"
done

# ── Compile every kernel, by running the riscv64 binaries under qemu-user ────────────────────
# teenyc (an x86 binary) runs natively when qemu-user execs it.
rm -rf "$stage"
mkdir -p "$stage" "$cache"
for t in "${tests[@]}"; do
    # Newest first, and not the .d depfile beside it: cargo leaves one directory per
    # build hash, so an alphabetical pick silently stages a stale binary.
    bin=$(ls -t "$target_dir/$triple/release/build/teeny-llm-core/"*"/out/$t-"* \
        | grep -v '\.d$' | head -1)
    TEENYC_CACHE_DIR=$cache TEENYC_CAPABILITY=$capability INKLING2_FIXTURES=$fixtures \
        "$qemu" -cpu "$qemu_cpu" -L $sysroot "$bin" --test-threads=1 >/dev/null
    cp "$bin" "$stage/$t"
done
cp -r "$cache" "$stage/kernel-cache"
cp -r "$fixtures" "$stage/fixtures"
echo "staged: $stage ($(du -sh "$stage" | cut -f1))"

[[ -z "$host" ]] && exit 0

# ── Copy to the board and run ────────────────────────────────────────────────────────────────
ssh "$host" "mkdir -p $remote"
rsync -a --delete "$stage/" "$host:$remote/"
for t in "${tests[@]}"; do
    ssh "$host" "cd $remote && uname -m && \
        TEENYC_PATH=/nonexistent/teenyc TEENYC_CACHE_DIR=\$PWD/kernel-cache \
        TEENYC_CAPABILITY=$capability INKLING2_FIXTURES=\$PWD/fixtures ./$t --test-threads=1"
done
