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

The demo tools run for real: `get_weather` fetches current conditions from
[Open-Meteo](https://open-meteo.com) (needs `curl` and internet on the board; the city names
you ask about go to open-meteo.com), `set_timer` counts down and prints when it is done, and
`send_message` appends the message to `~/teeny-gemma-outbox.txt` (there is no messaging service
to send through). Each call and its result are printed. For your own tools (`--tools
<file.json>`, OpenAI style), or with `--manual-tools`, you play the tool: type its result at the
`<tool> result>` prompt and the model answers from it. `/reset` starts a new conversation and
`/quit` exits.

After each reply the chat prints its speed:

```text
[18 new prompt tokens in 1.48 s (12.1 tokens/s); first token after 1.48 s; 12 tokens out at 4.8 tokens/s (4.4 decode); context 251/1024]
```

- **prompt tokens/s**: how fast the new part of the prompt is processed;
- **first token after**: the wait before the reply starts appearing;
- **decode**: the streaming speed, one token per step after the first.

Measured on a SpacemiT K3 with all 8 X100 cores (the default):

| | First turn (209-token prompt with tool declarations) | Follow-up turn |
|---|---|---|
| Prompt | 66.7 tokens/s | 36.1 tokens/s |
| First token after | 3.1 s | 0.53 s |
| Decode | 11.3 tokens/s | 11.3 tokens/s |

How it got there (first-turn prompt / first token / decode):

| Change | Prompt | First token after | Decode |
|---|---|---|---|
| Start: 128-lane linear blocks, 256-lane softmax state | 18.9 tokens/s | 11.0 s | 4.5 tokens/s |
| 16-lane linear blocks on RISC-V (no register spills) | 25.4 tokens/s | 8.2 s | 5.1 tokens/s |
| Attention: one-element softmax state, window bounds | 34.9 tokens/s | 6.0 s | 6.5 tokens/s |
| Linear weights reused from cache across a chunk's rows | 43.5 tokens/s | 4.8 s | 6.5 tokens/s |
| Persistent launcher thread pool (no thread start per kernel) | 66.7 tokens/s | 3.1 s | 11.3 tokens/s |

A decode step now takes about 90 ms. Before the thread pool, `TEENY_GEMMA_TRACE=1` (which
prints every kernel's time and each step's total) showed lm_head at 42 ms, attention at 14 ms
and host-side copies at 22 ms of a 170 ms step; a step launches about 390 kernels, and starting
threads for each was most of the rest. `examples/matmul_bench.rs` times the linear kernel alone,
for a list of block sizes and both program orderings.

Each kernel launch spreads its program ids over a pool of threads, one per available core;
`TEENY_RISCV_THREADS=<n>` overrides the count. With the same chat, measured with 128-lane
blocks and no pool (board load average about 2, so rough):

| Threads | First-turn prompt | Decode | Follow-up first token after |
|---|---|---|---|
| 1 | 3.1 tokens/s | 1.3 tokens/s | 9.2 s |
| 2 | 5.8 tokens/s | 2.3 tokens/s | 4.9 s |
| 4 | 11.2 tokens/s | 3.8 tokens/s | 2.6 s |
| 8 | 18.7 tokens/s | 4.4 tokens/s | 1.75 s |

Prompt processing scales almost linearly; decode levels off past 4 threads.

#### The A100 AI cores

The K3's other 8 cores (CPUs 8–15) are A100 "AI" cores: in-order, 1.8 GHz, 1024-bit vectors.
Processes start on the X100 cores, and `taskset -c 8-15` fails with "Invalid argument". To move
a process there, write its pid to SpacemiT's `/proc/set_ai_thread`; its children inherit it:

```bash
ssh -t board 'cd teeny-gemma-chat && bash -c "echo \$\$ > /proc/set_ai_thread; TEENYC_PATH=/nonexistent/teenyc TEENYC_CACHE_DIR=\$PWD/kernel-cache TEENYC_CAPABILITY=spacemit-k3 exec ./chat --model model"'
```

The answers are the same, but slower (8 threads each, before the thread pool):

| | X100 | A100 |
|---|---|---|
| First-turn prompt | 43.5 tokens/s | 13.9 tokens/s |
| First token after | 4.8 s | 14.9 s |
| Decode | 6.5 tokens/s | 3.1 tokens/s |
| Decode step with `TEENY_GEMMA_TRACE=1`: total / attention / lm_head | 169 / 14 / 42 ms | 345 / 44 / 87 ms |

Before the attention change the A100 decoded at 0.6 tokens/s: attention took 1.5 s of its
1.8 s step, mostly 256-lane `exp` calls, which an in-order core cannot hide. So the X100 cores
stay the default. Kernels are compiled for 128-bit vectors and f32, which uses a quarter of the
A100's 1024-bit registers and none of its IME matrix unit; that is where the A100 would have to
gain.

#### All 16 cores in one chat

`TEENY_RISCV_AI_THREADS=<n>` moves the launcher pool's last `n` threads to the A100 cores at
startup, so one chat uses both clusters; slower A100 threads simply take fewer chunks of work:

```bash
ssh -t board 'cd teeny-gemma-chat && TEENY_RISCV_THREADS=16 TEENY_RISCV_AI_THREADS=8 TEENYC_PATH=/nonexistent/teenyc TEENYC_CACHE_DIR=$PWD/kernel-cache TEENYC_CAPABILITY=spacemit-k3 ./chat --model model'
```

| | 8 X100 cores | 16 cores (8 X100 + 8 A100) |
|---|---|---|
| First-turn prompt | 66.7 tokens/s | 61.4 tokens/s |
| First token after | 3.1 s | 3.4 s |
| Decode | 11.3 tokens/s | 12.6 tokens/s |

Decode gains about 11%; prompt processing loses a little. Both clusters share one memory bus
(about 21 GB/s for either alone, 24 GB/s together), and decode reads every weight for every
token, so extra cores add little.

## On a CUDA machine

```bash
cargo run --release -p teeny-gemma --example chat -- --model <model folder>
```

`--example functiongemma` answers a single tool-calling request instead.

## License

Apache-2.0. See [LICENSE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE).
