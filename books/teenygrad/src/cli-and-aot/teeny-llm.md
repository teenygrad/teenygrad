# `teeny-llm`

`teeny-llm` is the workspace's LLM inference stack, aiming for vLLM-style serving with 2x the
throughput (see the [roadmap](../appendix/faq-and-roadmap.md)). It is split into three crates
under `apps/teeny-llm/`:

| Crate              | Kind      | Role                                                                    |
| ------------------ | --------- | ----------------------------------------------------------------------- |
| `teeny-llm-core`   | library   | The API traits models implement (chat, completions, embeddings), the model factory, and the `Engine` frontends drive |
| `teeny-llm-server` | lib + bin | OpenAI-compatible HTTP API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`) |
| `teeny-llm-chat`   | binary    | Interactive terminal chat, running the engine in-process                |

Each binary serves the models it is built with; a model is a cargo feature, with the backend it
runs on. `--model` takes a model by tag (`functiongemma3`, or `functiongemma3:270m`), loads its
weights from the Hugging Face cache (or `--model-dir`), and serves it through `ModelEngine`,
which routes each request to the API the model implements. `--echo` (server) and no `--model`
(chat) use the echo models instead, which need no weights.

```bash
cargo install teeny-llm-server teeny-llm-chat --features functiongemma3,cuda
teeny-llm-server --model functiongemma3 --port 8000   # OpenAI-compatible API
teeny-llm-chat --model functiongemma3 --tools tools.json
```

> **TODO**: fill this chapter in as the stack gains real functionality: model loading,
> batching/scheduling behavior.
