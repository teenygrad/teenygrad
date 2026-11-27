# `teeny-llm`

`teeny-llm` is the workspace's LLM inference stack, aiming for vLLM-style serving with 2x the
throughput (see the [roadmap](../appendix/faq-and-roadmap.md)). It is split into crates under
`apps/`:

| Crate       | Kind      | Role |
| ----------- | --------- | ---- |
| `teeny-llm` | library   | The LLM API traits models implement (chat, completions, embeddings, moderation), and the model factory (`Model::load`) |
| `teeny-audio` | library | The audio API traits (transcription, speech) and WAV/PCM decoding and encoding |
| `teeny-image` | library | The image generation API trait (generations, edits, variations) and PNG decoding and encoding |
| `teeny-api` | lib + bin | OpenAI-compatible HTTP API, one feature per domain (`llm` by default, `audio`, `image`), with the `Engine` its routes drive and the `ModelEngine` serving each domain's models, and a `/v1/events` websocket of what the server is doing |
| `teeny-bench` | binary (`testing/`) | llama-bench-style benchmarks: prompt and decode tokens/s, time to first token, decode latency percentiles, in-process or through `teeny-api` |
| `teeny-webapp` | binary | A browser UI for `teeny-api`, embedded: a chat with tool calls, and a console of `/v1/events` |

The binary serves the models it is built with; a model is a cargo feature, with the backend it
runs on. `--model` takes a model by tag (`functiongemma3`, or `functiongemma3:270m`), loads its
weights from the Hugging Face cache (or `--model-dir`), and serves it through `ModelEngine`,
which routes each request to the API the model implements. `--echo` serves the echo models
instead, which need no weights.

```bash
cargo install teeny-api --features functiongemma3,cuda
teeny-api --model functiongemma3 --port 8000   # OpenAI-compatible API
cargo run -p teeny-webapp -- --api-url http://127.0.0.1:8000   # UI at http://127.0.0.1:8080
```

The server listens before its models load, so the webapp's console shows them loading.

Every chat and text completion response carries the model's own `timings` (llama.cpp's
format: prompt and decode tokens/s, cached tokens, time to first token, each decode step's
latency), which the webapp shows and `teeny-bench` aggregates:

```bash
cargo run --release -p teeny-bench --features functiongemma3,cuda -- --model functiongemma3 -p 64,256,512 -n 128
```

> **TODO**: fill this chapter in as the stack gains real functionality: model loading,
> batching/scheduling behavior.
