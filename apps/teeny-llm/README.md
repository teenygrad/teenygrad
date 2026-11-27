# teeny-llm

Core LLM inference library for [teenygrad](https://teenygrad.org). It defines the
transport-independent pieces that
[`teeny-api`](https://docs.teenygrad.org/api/teenygrad/teeny_api/) (OpenAI-compatible
HTTP API) serves:

- one module per OpenAI API, each with its request and response types, the capabilities a
  model declares, `validate` (refusing what a model cannot do, by parameter), the trait models
  implement, and an echo model as its reference implementation: `chat` (`ChatModel`),
  `completion` (`CompletionModel`), `embedding` (`EmbeddingModel`),
  `moderation` (`ModerationModel`)
- `timings`: `Timings`, how long a generation took, phase by phase (queue, prompt tokens run
  and reused, first token, decode, and each decode step's latency), which chat and completion
  models send just before their usage; `Recorder` measures it
- `tokenizer`: the `Tokenizer` trait, with a byte tokenizer and Hugging Face's (`hf-tokenizer`)
- `models::Model`: the models a build includes, by tag (`functiongemma3:270m`), and
  `Model::load`, which loads one as a `LoadedModel` exposing the APIs it serves

## Models

Model implementations live in `teeny_llm::models`, one module per model family, each behind
a cargo feature of the same name, plus a backend feature (`cuda`, `riscv` or `qemu`). The default
features are `functiongemma3` and `cuda`; for RISC-V use
`--no-default-features --features functiongemma3,riscv`.

| Model | Feature | Module | Guide |
|---|---|---|---|
| FunctionGemma (Gemma 3 270M) | `functiongemma3` | `models::functiongemma` | [docs/functiongemma.md](docs/functiongemma.md) |

Kernels a model shares with others are in `teeny_llm::kernels` (grouped-query attention over a
KV cache, a portable bias-free linear, RoPE, last-row select); a model's own kernels are in its
module.

```bash
cargo test -p teeny-llm --test test_llm_kernels
cargo run --release -p teeny-llm --example chat -- --model functiongemma3   # weights from the HF cache, or --model-dir
```

Loaded through `Model::load`, the models are served by `teeny-api`
(built with the model's feature: `--features functiongemma3,cuda`). The `chat`
example drives FunctionGemma directly: `--mode chat` (the default) in the terminal,
or `--mode tool --prompt ...` for the function calls it picks for one request. For a browser,
run teeny-api with teeny-webapp.

## License

Apache-2.0. See [LICENSE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE).
