#!/usr/bin/env bash
# Packages FunctionGemma's example (teeny-llm-core's `chat`, in chat mode) for a
# RISC-V board (e.g. a SpacemiT K3) and, given a host, copies it there and starts an
# interactive chat over ssh.
#
# The board needs nothing installed beyond glibc: this script cross-compiles the `chat` example for
# riscv64gc-unknown-linux-gnu, compiles every kernel here (where teenyc lives) by starting that
# same binary under qemu-user (the chat compiles its step graphs before reading any input, so it
# is fed `/quit`), then stages the binary, the kernel cache and the model folder. On the board it
# runs with teenyc disabled, so a cache miss fails loudly instead of trying to compile.
#
# Usage: apps/teeny-llm/teeny-llm-core/scripts/functiongemma-board-chat.sh [-H SSH_HOST] [--skip-build] [-- CHAT_ARGS...]
#   Without -H, only builds and stages (in target/riscv64-native/gemma-stage).
#   Needs: rustup target riscv64gc-unknown-linux-gnu, gcc-riscv64-linux-gnu, qemu-user(-static),
#          rsync; teenyc (TEENYC_PATH, or found by teeny-compiler).
#   Env:   GEMMA_MODEL_DIR (default: the Hugging Face cache snapshot of
#          google/functiongemma-270m-it); TEENYC_CAPABILITY (default spacemit-k3);
#          REMOTE_DIR (default functiongemma-chat, relative to the board user's home).
set -euo pipefail

host=""
skip_build=0
chat_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -H) host="$2"; shift 2 ;;
        --skip-build) skip_build=1; shift ;;
        --) shift; chat_args=("$@"); break ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

cd "$(dirname "$0")/../../../.."
capability=${TEENYC_CAPABILITY:-spacemit-k3}
remote=${REMOTE_DIR:-functiongemma-chat}
model=${GEMMA_MODEL_DIR:-$(ls -d "$HOME"/.cache/huggingface/hub/models--google--functiongemma-270m-it/snapshots/* 2>/dev/null | head -1)}
[[ -n "$model" && -f "$model/config.json" ]] || {
    echo "model folder not found: set GEMMA_MODEL_DIR to the folder with config.json" >&2
    exit 1
}
triple=riscv64gc-unknown-linux-gnu
target_dir=target/riscv64-native
bin=$target_dir/$triple/release/examples/chat
stage=$target_dir/gemma-stage
cache=$PWD/$target_dir/gemma-kernel-cache
sysroot=/usr/riscv64-linux-gnu
qemu=$(command -v qemu-riscv64 || command -v qemu-riscv64-static) || {
    echo "qemu-riscv64 not found (apt install qemu-user-static)" >&2
    exit 1
}

if [[ $skip_build == 0 ]]; then
    CARGO_TARGET_RISCV64GC_UNKNOWN_LINUX_GNU_LINKER=riscv64-linux-gnu-gcc \
        CARGO_TARGET_DIR=$target_dir \
        cargo build --release --target $triple -p teeny-llm-core --example chat \
        --no-default-features --features functiongemma3,riscv
fi

# ── Stage: binary and model (dereferencing the Hugging Face cache's symlinks) ─────────────────
rm -rf "$stage"
mkdir -p "$stage/model" "$cache"
cp "$bin" "$stage/chat"
for f in config.json model.safetensors tokenizer.json chat_template.jinja special_tokens_map.json; do
    cp -L "$model/$f" "$stage/model/"
done

# ── Compile every kernel, by starting the riscv64 binary itself under qemu-user ──────────────
# teenyc (an x86 binary) runs natively when qemu-user execs it. Kernels already in the cache
# aren't recompiled.
echo "building the kernel cache under qemu (loads the model first; a few minutes)..."
(
    cd "$stage"
    echo /quit | TEENYC_CACHE_DIR=$cache TEENYC_CAPABILITY=$capability \
        "$qemu" -cpu max -L $sysroot ./chat --model-dir model "${chat_args[@]}"
)
cp -r "$cache" "$stage/kernel-cache"
echo "staged: $stage ($(du -sh "$stage" | cut -f1))"

[[ -z "$host" ]] && exit 0

# ── Copy to the board and chat ────────────────────────────────────────────────────────────────
ssh "$host" "mkdir -p $remote"
rsync -a --delete --info=progress2 "$stage/" "$host:$remote/"
printf -v args_q '%q ' "${chat_args[@]}"
ssh -t "$host" "cd $remote && uname -m && nproc && \
    TEENYC_PATH=/nonexistent/teenyc TEENYC_CACHE_DIR=\$PWD/kernel-cache \
    TEENYC_CAPABILITY=$capability ./chat --model-dir model $args_q"
