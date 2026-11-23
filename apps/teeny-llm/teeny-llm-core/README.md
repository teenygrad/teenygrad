# teeny-llm-core

Core LLM inference library for [teenygrad](https://teenygrad.org). It defines the
transport-independent pieces shared by
[`teeny-llm-server`](https://docs.teenygrad.org/api/teenygrad/teeny_llm_server/) (OpenAI-compatible
HTTP API) and [`teeny-llm-chat`](https://docs.teenygrad.org/api/teenygrad/teeny_llm_chat/)
(interactive CLI):

- `ChatMessage` / `Role`: a conversation
- `Prompt`, `SamplingParams`, `GenerationRequest`: what to generate
- `Engine`: the trait an inference backend implements (streaming generation, optional embeddings)
- `EchoEngine`: a placeholder engine that echoes the last user message, so the server and CLI can be
  exercised end to end before real model execution lands

## Models

Model implementations live in `teeny_llm_core::models`, one module per model family, each behind
a cargo feature of the same name, plus a backend feature (`cuda`, `riscv` or `qemu`). The default
features are `functiongemma` and `cuda`; for RISC-V use
`--no-default-features --features functiongemma,riscv`.

| Model | Feature | Module | Guide |
|---|---|---|---|
| FunctionGemma (Gemma 3 270M) | `functiongemma` | `models::functiongemma` | [docs/functiongemma.md](docs/functiongemma.md) |

Kernels a model shares with others are in `teeny_llm_core::kernels` (grouped-query attention over a
KV cache, a portable bias-free linear, RoPE, last-row select); a model's own kernels are in its
module.

```bash
cargo test -p teeny-llm-core --test test_llm_kernels
cargo run --release -p teeny-llm-core --example functiongemma -- --model <model folder>
```

> ⚠️ The models are not yet served through `Engine`: `teeny-llm-server` and `teeny-llm-chat`
> still run `EchoEngine`. Use the `functiongemma` example to chat with FunctionGemma (`--mode chat`,
> the default) or to see the function calls it picks for one request (`--mode tool --prompt ...`),
> and `functiongemma_serve` for the weather web demo.

## License

Apache-2.0. See [LICENSE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE).
