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

Unknown request fields are ignored. Fields that change the semantics in ways not supported yet
(`n > 1`, `echo`, `suffix`, non-text content parts, `encoding_format: "base64"`) are rejected with
an OpenAI-style `400` error, not silently dropped.

> ⚠️ Early scaffolding: the server currently runs the placeholder `EchoEngine`
> (model `teeny-echo`), which echoes the last user message.

## Getting started

```bash
cargo install teeny-llm-server
teeny-llm-server --port 8000

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "teeny-echo", "messages": [{"role": "user", "content": "hello"}]}'
```

The Vue scaffolds in `webapps/` (chat UI, admin console) aren't wired up yet.

## License

Apache-2.0. See [LICENSE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE).
