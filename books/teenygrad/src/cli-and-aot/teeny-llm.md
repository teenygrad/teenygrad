# `teeny-llm`

`teeny-llm` is the workspace's LLM inference stack, aiming for vLLM-style serving with 2x the
throughput (see the [roadmap](../appendix/faq-and-roadmap.md)). It is split into three crates
under `apps/teeny-llm/`:

| Crate              | Kind      | Role                                                                    |
| ------------------ | --------- | ----------------------------------------------------------------------- |
| `teeny-llm-core`   | library   | Chat/sampling types and the `Engine` trait inference backends implement |
| `teeny-llm-server` | lib + bin | OpenAI-compatible HTTP API (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`) |
| `teeny-llm-chat`   | binary    | Interactive terminal chat, running the engine in-process                |

> ⚠️ **Early scaffolding.** No real model execution yet: both binaries run the placeholder
> `EchoEngine` (model `teeny-echo`), which echoes the last user message back.

```bash
cargo install teeny-llm-server teeny-llm-chat
teeny-llm-server --port 8000     # OpenAI-compatible API
teeny-llm-chat                   # interactive chat
```

> **TODO**: fill this chapter in as the stack gains real functionality: model loading,
> batching/scheduling behavior.
