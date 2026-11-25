# teeny-llm-server

OpenAI-compatible HTTP API for [teenygrad](https://teenygrad.org) LLM inference, built on
[`teeny-llm-core`](https://docs.teenygrad.org/api/teenygrad/teeny_llm_core/). Its request and
response types follow the OpenAI OpenAPI spec (v2.3.0), so the official OpenAI SDKs and other
OpenAI-compatible clients can talk to it.

| Method | Path                          | Notes                                    |
| ------ | ----------------------------- | ---------------------------------------- |
| GET    | `/v1/models`                  |                                          |
| GET    | `/v1/models/{model}`          |                                          |
| POST   | `/v1/chat/completions`        | `stream: true` → server-sent events      |
| POST   | `/v1/completions`             | `stream: true` → server-sent events      |
| POST   | `/v1/embeddings`              | `501` until the engine has an embedder   |
| GET    | `/health`                     |                                          |

The chat API is served whole: tools and tool calls (streamed as deltas), `tool_choice`,
content parts, `n`, `logprobs`, `response_format`, `stream_options.include_usage`. What the
chosen model cannot do (images for a text-only model, `n > 1`, sampling for a greedy one) is
refused with an OpenAI-style `400` naming the parameter, never silently dropped. Fields that
concern only the service (`store`, `service_tier`, ...) are ignored.

## Getting started

The binary serves the models it is built with. Each model is a feature, with the backend it
runs on:

```bash
cargo install teeny-llm-server --features functiongemma3,cuda
teeny-llm-server --model functiongemma3 --port 8000

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "functiongemma3", "messages": [{"role": "user", "content": "hello"}]}'
```

`--model <tag>` (repeatable) picks the models; the weights come from the local Hugging Face
cache, or `--model-dir` for a single model, and `--context` sets the KV-cache size. Without
a model feature, `--echo` serves `teeny-echo`, which echoes the last user message, for trying
the API without weights. Legacy `/v1/completions` and `/v1/embeddings` answer once models
implement them.

The Vue scaffolds in `webapps/` (chat UI, admin console) aren't wired up yet.

## License

Apache-2.0. See [LICENSE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE).
