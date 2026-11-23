# Reproducing the FunctionGemma results on a SpacemiT K3

Every number in [functiongemma.md](functiongemma.md) comes from one of the commands below. They were measured on
a SpacemiT K3 Pico ITX (Bianbu 4.0.4, kernel 6.18.3, 16 cores, 31 GB) from branch
`feature/functiongemma`; the host was x86-64 Linux with an RTX 5070 Ti for the CUDA checks.

Throughout, `board` is your ssh alias for the K3 and commands run from the workspace root.

## 0. Setup

Follow steps 1 and 2 of [functiongemma.md](functiongemma.md) (host prerequisites, the gated model download, passwordless ssh). Then
build, copy and try the chat once:

```bash
apps/teeny-llm/teeny-llm-core/scripts/functiongemma-board-chat.sh -H board
```

Type `/quit` to leave. This stages everything to `~/functiongemma-chat` on the board; all board
commands below run from there. Later builds: `board-chat.sh` without `-H` stages into
`target/riscv64-native/gemma-stage`, then

```bash
rsync -a --delete target/riscv64-native/gemma-stage/ board:functiongemma-chat/
```

Every run on the board needs these variables. Set this on the host; the single quotes keep
`$PWD` for the board's shell to expand, and the commands below splice it in as `'"$E"'`:

```bash
E='TEENYC_PATH=/nonexistent/teenyc TEENYC_CACHE_DIR=$PWD/kernel-cache TEENYC_CAPABILITY=spacemit-k3'
```

Measure on an otherwise idle board (check `uptime`); the clusters share one memory bus, so
anything else running changes the numbers.

## 1. Correctness

**CUDA against PyTorch (FP64).** Fixtures need torch and transformers once:

```bash
python apps/teeny-llm/teeny-llm-core/tests/fixtures/functiongemma/gen_reference.py --out ~/.cache/teeny-llm/fixtures/functiongemma
GEMMA_MODEL_DIR=$(ls -d ~/.cache/huggingface/hub/models--google--functiongemma-270m-it/snapshots/*/ | head -1) \
GEMMA_FIXTURES=~/.cache/teeny-llm/fixtures/functiongemma \
  cargo test --release -p teeny-llm-core --test test_functiongemma -- --ignored --nocapture
```

Expect all 24 greedy tokens to match and `worst logit difference over 24 steps: 9.3e-5`.

**The RISC-V kernels against PyTorch, under qemu** (about 3 minutes for 6 steps; the test looks
for `qemu-riscv64` unless told otherwise):

```bash
TEENYC_QEMU_RISCV64=$(command -v qemu-riscv64-static) TEENYC_CAPABILITY=spacemit-k3 GEMMA_STEPS=6 \
GEMMA_MODEL_DIR=$(ls -d ~/.cache/huggingface/hub/models--google--functiongemma-270m-it/snapshots/*/ | head -1) \
GEMMA_FIXTURES=~/.cache/teeny-llm/fixtures/functiongemma CARGO_TARGET_DIR=target/qemu-test \
  cargo test --release -p teeny-llm-core --no-default-features --features functiongemma,qemu \
    --test test_functiongemma -- --ignored --nocapture test_functiongemma_matches_pytorch
```

Expect 6 matching tokens and a worst difference of about 9e-5.

**Past the 512-token sliding window.** The PyTorch prompt is short, so this checks long chats: 12
turns with a fixed tool result, ending at 1007 tokens.

```bash
R='{"city":"Smolensk, Russia","conditions":"rain","humidity_percent":94,"temperature_c":2.0,"wind_kmh":14.7}'
{ for c in Smolensk Vyazma Moscow Tula Kursk Orel; do echo "What is the weather in $c?"; echo "$R"; done; echo /quit; } > long.txt
scp long.txt board:
ssh board 'cd functiongemma-chat && env '"$E"' ./chat --model model --manual-tools < ~/long.txt 2>&1 | grep -o "[0-9]* tokens out\|context [0-9]*"'
```

Expect replies of 17, 55, 17, 56, 15, 54, 16, 55, 15, 55, 16, 56 tokens and a final context of
1007, the same as the CUDA build (`target/release/examples/functiongemma --model <model folder>
--manual-tools < long.txt`). Before the fix for teenyc bug 5 the RISC-V build produced `<pad>`
from the turn after 498 tokens.

## 2. Chat speed

The benchmark conversation is two turns: a question that triggers a tool call, then a fixed tool
result. `--manual-tools` keeps the tool result fixed (the live weather would change the text):

```bash
ssh board 'cd functiongemma-chat && printf "What is the weather in Paris?\nsunny, 21C\n/quit\n" | env '"$E"' ./chat --model model --manual-tools'
```

Each turn prints `[N new prompt tokens in … (… tokens/s); first token after … s; M tokens out at
… (… decode) …]`. functiongemma.md's first-turn row is the first line, the follow-up row the second.
Both turns end with the same answer, "The current weather in Paris is sunny, with a temperature of
21°C (74°F)."

The three core setups (same command, different environment):

| Setup | Add to the environment | Expected first turn: prompt / first token / decode |
|---|---|---|
| 8 X100 cores (default) | nothing | 66.7 tokens/s / 3.1 s / 11.3 tokens/s |
| 16 cores | `TEENY_RISCV_THREADS=16 TEENY_RISCV_AI_THREADS=8` | 61.4 tokens/s / 3.4 s / 12.6 tokens/s |
| 8 A100 cores only | run under `bash -c 'echo $$ > /proc/set_ai_thread; exec …'` | slower; see functiongemma.md's A100 table (measured before the thread pool) |

To confirm where threads run, while a chat is waiting for input after its first message:

```bash
ssh board 'p=$(pgrep -n -x chat); for t in /proc/$p/task/*; do grep -h Cpus_allowed_list $t/status; done | sort | uniq -c'
```

With 16 cores expect 9 threads on `0-7` (the main thread and 8 pool threads) and 8 on `8-15`.

**The "how it got there" table.** Each row is a commit on `feature/functiongemma`; check it out,
rebuild with `board-chat.sh`, copy, and run the benchmark conversation:

| Row | Commit |
|---|---|
| 16-lane linear blocks on RISC-V | `39d0ab7db` |
| Attention: one-element softmax state | `46c6b2431` |
| Linear weights reused across a chunk's rows | `82f962a15` |
| Persistent launcher thread pool | `8ea096432` |

(The first row, 128-lane blocks, is the commit before `39d0ab7db`.) Commits before `03ce26907`
have no `--manual-tools` flag and always ask you for the tool result, so drop the flag there.

## 3. Where a decode step's time goes

`TEENY_GEMMA_TRACE=1` prints every kernel's time (`[i/387] node … name [shape]: time`) and each
step's total (`[step 1 tokens] total …`); it slows a step somewhat. To average the one-token
steps:

```bash
ssh board 'cd functiongemma-chat && printf "What is the weather in Paris?\nsunny, 21C\n/quit\n" | env '"$E"' TEENY_GEMMA_TRACE=1 ./chat --model model --manual-tools 2> trace.log >/dev/null; grep -c "^\[step 1 tokens\]" trace.log'
scp board:functiongemma-chat/trace.log .
python3 - trace.log <<'PY'
import re, sys, collections
unit = {"s": 1e3, "ms": 1, "µs": 1e-3, "ns": 1e-6}
def ms(v):
    m = re.match(r"([\d.]+)(\D+)", v)
    return float(m[1]) * unit[m[2]]
steps, per_op, buf = [], collections.Counter(), []
for line in open(sys.argv[1]):
    if m := re.match(r"\[\d+/\d+\] node \d+ (\S+) .*: (\S+)$", line.strip()):
        buf.append((re.sub(r"model\.layers\.\d+\.", "layer.", m[1]), ms(m[2])))
    elif m := re.match(r"\[step (\d+) tokens\] total (\S+)", line.strip()):
        if m[1] == "1":
            steps.append(ms(m[2]))
            for k, t in buf:
                per_op[k] += t
        buf = []
n = len(steps)
print(f"{n} decode steps, {sum(steps) / n:.1f} ms each")
for k, t in per_op.most_common(8):
    print(f"  {k:40s} {t / n:7.2f} ms")
PY
```

## 4. Memory bandwidth

The "shared memory bus" numbers (about 21 GB/s for either cluster alone, 24 GB/s together) come
from this stand-alone program, not part of the workspace:

```rust
// bw.rs: read bandwidth with u64 loads. Usage: bw <threads> <seconds>
use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (t, secs): (usize, f64) = (a[1].parse().unwrap(), a[2].parse().unwrap());
    let n = 128 << 20; // 1 GiB of u64
    let v: Vec<u64> = (0..n as u64).collect();
    let (start, mut bytes) = (Instant::now(), 0usize);
    while start.elapsed().as_secs_f64() < secs {
        std::thread::scope(|sc| {
            let hs: Vec<_> = v.chunks(n / t).map(|c| sc.spawn(move || {
                let mut x = [0u64; 8];
                for b in c.chunks_exact(8) { for i in 0..8 { x[i] ^= b[i]; } }
                x.iter().fold(0, |p, q| p ^ q)
            })).collect();
            for h in hs { std::hint::black_box(h.join().unwrap()); }
        });
        bytes += n * 8;
    }
    println!("{t} threads: {:.1} GB/s", bytes as f64 / start.elapsed().as_secs_f64() / 1e9);
}
```

```bash
rustc -O --edition 2021 --target riscv64gc-unknown-linux-gnu -C linker=riscv64-linux-gnu-gcc bw.rs -o bw && scp bw board:
ssh board './bw 8 8; bash -c "echo \$\$ > /proc/set_ai_thread; exec ./bw 8 8"'
ssh board './bw 8 12 > x.out & bash -c "echo \$\$ > /proc/set_ai_thread; exec ./bw 8 12" > a.out; wait; cat x.out a.out'
```

The first line gives the X100 and A100 clusters alone, the second both at once (add the two).

## 5. Temperature

None of the runs throttled (72–80 °C, full clocks). To check during a run:

```bash
ssh board 'for z in 3 4 5 6; do echo "$(cat /sys/class/thermal/thermal_zone$z/type) $(( $(cat /sys/class/thermal/thermal_zone$z/temp) / 1000 )) C"; done
  echo "MHz: $(( $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) / 1000 )) / $(( $(cat /sys/devices/system/cpu/cpu8/cpufreq/scaling_cur_freq) / 1000 ))"
  echo "throttle states: $(cat /sys/class/thermal/cooling_device2/cur_state) / $(cat /sys/class/thermal/cooling_device3/cur_state)"'
```

Expect 2200 / 1800 MHz and throttle states 0 / 0 (X100 / A100).
