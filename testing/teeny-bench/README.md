# teeny-bench

[llama-bench](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench)-style
benchmarks of [teenygrad](https://teenygrad.org)'s chat models, from the models' own
measurements (`teeny_llm::Timings`):

- `pp<N>` runs an N-token prompt and generates one token: prompt tokens per second and the
  time to first token;
- `tg<N>` generates N tokens after a short prompt: decode tokens per second and each decode
  step's latency (p50, p90, p99 over every run).

It benchmarks a model in-process (`--model`, loaded with `Model::load`) or one a running
[`teeny-api`](../../apps/teeny-api) serves (`--api-url`, reading the llama.cpp-format
`timings` it returns, and recording each round trip as well). Both report the same numbers.

## Running it

Build in release, with the model and backend features, as for teeny-api:

```bash
cargo run --release -p teeny-bench --features functiongemma3,cuda -- --model functiongemma3
cargo run --release -p teeny-bench -- --api-url http://127.0.0.1:8000     # any served model
cargo run --release -p teeny-bench -- --echo                              # try the tool
```

| Flag | Default | |
| ---- | ------- | - |
| `-p, --prompt` | `64,256` | prompt lengths to test, in tokens |
| `-n, --gen` | `64` | generation lengths to test |
| `-r, --repetitions` | `5` | measured runs per test |
| `--warmup` | `1` | unmeasured runs per test first |
| `-o, --output` | `md` | `md` (a table) or `json` (everything, each run included) |
| `--model-dir`, `--context` | | as for teeny-api |
| `--api-model` | first listed | the model to ask teeny-api for |

For example, FunctionGemma on an NVIDIA GeForce RTX 5070 (CUDA, release build):

**functiongemma3:270m (in-process, cuda)**, loaded in 3.9 s; 5 run(s) per test after 1 warm-up

| test | prompt | cached | generated | tokens/s | first token | p50 | p90 | p99 |
| ---- | -----: | -----: | --------: | -------: | ----------: | --: | --: | --: |
| pp64 | 64 | 9 | 1 | 937.7 ± 62.1 | 69.1 ms |  |  |  |
| pp256 | 256 | 9 | 1 | 900.5 ± 20.2 | 285.2 ms |  |  |  |
| pp512 | 512 | 9 | 1 | 849.9 ± 14.1 | 603.2 ms |  |  |  |
| tg64 | 18 | 9 | 64 | 80.5 ± 0.6 | 44.2 ms | 12.3 ms | 12.8 ms | 14.0 ms |
| tg128 | 18 | 9 | 128 | 78.4 ± 0.6 | 42.5 ms | 12.7 ms | 13.4 ms | 14.8 ms |

## How it measures

- **Prompt lengths are exact.** The chat template adds tokens of its own, and over the API
  there is no tokenizer to count with, so a benchmark first sizes its prompts with probe
  requests, then corrects each `pp` test against what the model reports running. Prompts
  shorter than the template's own tokens are refused. FunctionGemma runs prompts in 16-token
  chunks and any remainder a token at a time, so lengths that are multiples of 16 measure its
  real throughput: a 71-token prompt runs at half the rate of a 64-token one.
- **The cache stays out of it.** Every request opens with a run tag of its own, so the model's
  KV cache reuses only the template's opening (the `cached` column) and the prompt tokens
  are run, not recalled.
- **Generation may stop early.** A model can end its reply before the test's length; the
  `generated` column says how many tokens it produced, and the rates are over those.
- **Decode rates are over decode steps.** The first token comes out of the prompt phase, so
  `tg` throughput is the `generated - 1` decode steps over their time.

Not published to crates.io: build it from the repository.

## License

Apache-2.0. See [LICENSE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE).
