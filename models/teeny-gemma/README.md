# teeny-gemma

Gemma 3 text models on teenygrad, starting with FunctionGemma
([`google/functiongemma-270m-it`](https://huggingface.co/google/functiongemma-270m-it), Gemma 3
270M fine-tuned for function calling). Runs on CUDA and on RISC-V RVV boards.

## Chat with FunctionGemma on a RISC-V board (SpacemiT K3)

The board needs nothing installed beyond glibc: no Rust, no `teenyc`. Everything is built on an
x86-64 Linux host, including every kernel, and copied over.

### 1. Host prerequisites (once)

```bash
rustup target add riscv64gc-unknown-linux-gnu
sudo apt install gcc-riscv64-linux-gnu qemu-user-static rsync
cargo teeny install-toolchain      # teenyc, the kernel compiler
```

The model is gated on Hugging Face: accept its licence on the model page, then download it.

```bash
pip install -U huggingface_hub
huggingface-cli login
huggingface-cli download google/functiongemma-270m-it
```

The script finds it in `~/.cache/huggingface/hub`; set `GEMMA_MODEL_DIR` to use another folder.

### 2. SSH access to the board

The script logs in over ssh several times, so `ssh <board> true` should work without a password
prompt (a key without a passphrase, or one loaded with `ssh-add`).

### 3. Build, copy and start the chat

From the workspace root:

```bash
models/teeny-gemma/scripts/board-chat.sh -H board
```

This:

1. cross-compiles the `chat` example for `riscv64gc-unknown-linux-gnu`;
2. compiles every kernel for the K3 by starting that binary under qemu-user (a few minutes the
   first time; later runs reuse `target/riscv64-native/gemma-kernel-cache`);
3. stages the binary, kernel cache and model (about 560 MB) and rsyncs them to
   `~/teeny-gemma-chat` on the board;
4. starts the chat there with `teenyc` disabled, so a missing kernel fails loudly.

Without `-H` it only builds and stages, in `target/riscv64-native/gemma-stage`. Other options:
`--skip-build`, and chat flags after `--` (for example `-- --context 2048 --max-tokens 256`).
`TEENYC_CAPABILITY` picks the kernel target (default `spacemit-k3`; `generic-rvv1.0` for other
RVV 1.0 boards).

Once it's on the board, start the chat again without rebuilding:

```bash
ssh -t board 'cd teeny-gemma-chat && TEENYC_PATH=/nonexistent/teenyc TEENYC_CACHE_DIR=$PWD/kernel-cache TEENYC_CAPABILITY=spacemit-k3 ./chat --model model'
```

### 4. Using the chat

The demo tools are `get_weather`, `set_timer` and `send_message` (`--tools <file.json>` for your
own, OpenAI style). When the model calls a tool, you play the tool: type its result at the
`<tool> result>` prompt and the model answers from it. `/reset` starts a new conversation and
`/quit` exits.

After each reply the chat prints its speed:

```text
[18 new prompt tokens in 1.48 s (12.1 tokens/s); first token after 1.48 s; 12 tokens out at 4.8 tokens/s (4.4 decode); context 251/1024]
```

- **prompt tokens/s**: how fast the new part of the prompt is processed;
- **first token after**: the wait before the reply starts appearing;
- **decode**: the streaming speed, one token per step after the first.

Measured on a SpacemiT K3 with all 8 X100 cores (the default) and no tuning:

| | First turn (209-token prompt with tool declarations) | Follow-up turn |
|---|---|---|
| Prompt | 18.8 tokens/s | 12.1 tokens/s |
| First token after | 11.1 s | 1.5 s |
| Decode | 4.5 tokens/s | 4.4 tokens/s |

Each kernel launch spreads its program ids over one thread per available core;
`TEENY_RISCV_THREADS=<n>` overrides the count. With the same chat (board load average about 2,
so rough):

| Threads | First-turn prompt | Decode | Follow-up first token after |
|---|---|---|---|
| 1 | 3.1 tokens/s | 1.3 tokens/s | 9.2 s |
| 2 | 5.8 tokens/s | 2.3 tokens/s | 4.9 s |
| 4 | 11.2 tokens/s | 3.8 tokens/s | 2.6 s |
| 8 | 18.7 tokens/s | 4.4 tokens/s | 1.75 s |

Prompt processing scales almost linearly; decode levels off past 4 threads. The K3's 8 A100 AI
cores (CPUs 8–15) are not used: the kernel refuses to schedule ordinary processes on them
(`taskset -c 8-15` fails with "Invalid argument").

## On a CUDA machine

```bash
cargo run --release -p teeny-gemma --example chat -- --model <model folder>
```

`--example functiongemma` answers a single tool-calling request instead.

## License

Apache-2.0. See [LICENSE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE).
