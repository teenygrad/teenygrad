# `teeny-llm`

`teeny-llm` is the workspace's LLM inference stack, aiming for vLLM-style serving with 2x the
throughput (see the [roadmap](../appendix/faq-and-roadmap.md)). It is split into crates under
`apps/`:

| Crate       | Kind      | Role |
| ----------- | --------- | ---- |
| `teeny-llm` | library   | The LLM API traits models implement (chat, completions, embeddings, moderation; images until it moves to its own crate), and the model factory (`Model::load`) |
| `teeny-audio` | library | The audio API traits (transcription, speech) and WAV/PCM decoding and encoding |
| `teeny-api` | lib + bin | OpenAI-compatible HTTP API, one feature per domain (`llm` by default, `audio`, `image`), with the `Engine` its routes drive and the `ModelEngine` serving each domain's models |

The binary serves the models it is built with; a model is a cargo feature, with the backend it
runs on. `--model` takes a model by tag (`functiongemma3`, or `functiongemma3:270m`), loads its
weights from the Hugging Face cache (or `--model-dir`), and serves it through `ModelEngine`,
which routes each request to the API the model implements. `--echo` serves the echo models
instead, which need no weights.

```bash
cargo install teeny-api --features functiongemma3,cuda
teeny-api --model functiongemma3 --port 8000   # OpenAI-compatible API
```

> **TODO**: fill this chapter in as the stack gains real functionality: model loading,
> batching/scheduling behavior.
