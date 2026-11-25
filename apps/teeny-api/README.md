# teeny-api

OpenAI-compatible HTTP API for [teenygrad](https://teenygrad.org) LLM inference, built on
[`teeny-llm`](https://docs.teenygrad.org/api/teenygrad/teeny_llm/). Its request and
response types follow the OpenAI OpenAPI spec (v2.3.0), so the official OpenAI SDKs and other
OpenAI-compatible clients can talk to it.

| Method | Path                          | Notes                                    |
| ------ | ----------------------------- | ---------------------------------------- |
| GET    | `/v1/models`                  |                                          |
| GET    | `/v1/models/{model}`          |                                          |
| POST   | `/v1/chat/completions`        | `stream: true` → server-sent events      |
| POST   | `/v1/completions`             | `stream: true` → server-sent events      |
| POST   | `/v1/embeddings`              | `encoding_format: "base64"` supported    |
| POST   | `/v1/audio/transcriptions`    | multipart, WAV; every `response_format` |
| POST   | `/v1/audio/translations`      | as transcriptions, into English          |
| POST   | `/v1/audio/speech`            | wav or pcm; `stream_format: "sse"`       |
| POST   | `/v1/moderations`             | `model` optional: the first classifier   |
| POST   | `/v1/images/generations`      | PNG as `b64_json`; `stream: true` → SSE  |
| POST   | `/v1/images/edits`            | multipart PNG image(s) and mask          |
| POST   | `/v1/images/variations`       | multipart PNG image                      |
| GET    | `/v1/events`                  | websocket of server events (see below)   |
| GET    | `/health`                     |                                          |

The chat API is served whole: tools and tool calls (streamed as deltas), `tool_choice`,
content parts, `n`, `logprobs`, `response_format`, `stream_options.include_usage`. What the
chosen model cannot do (images for a text-only model, `n > 1`, sampling for a greedy one) is
refused with an OpenAI-style `400` naming the parameter, never silently dropped. Fields that
concern only the service (`store`, `service_tier`, ...) are ignored.

## Features

Each domain is a feature, bringing in its endpoints and the crate its models come from
(`teeny-llm`, `teeny-audio`, `teeny-image`);
`/v1/models` and `/health` are always served:

| Feature | Endpoints |
| ------- | --------- |
| `llm` (default) | `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/moderations` |
| `audio` | `/v1/audio/transcriptions`, `/v1/audio/translations`, `/v1/audio/speech` |
| `image` | `/v1/images/generations`, `/v1/images/edits`, `/v1/images/variations` |

The crate owns the serving layer: `Engine`, the trait the routes drive, and `ModelEngine`,
which serves `ServedModel`s (each the API handles one domain crate's model implements, a
teeny-llm `LoadedModel` converting into one) and routes each request to the model and
API it names. Each domain crate's errors convert into `teeny_api::Error`, so they map to
HTTP in one place.

## Events

`/v1/events` is a websocket over which the server pushes what happens in it, for dashboards
and the webapp. Each event is one JSON text frame, with a `type`, a `time` (Unix
milliseconds) and its fields:

| `type` | Fields | When |
| ------ | ------ | ---- |
| `hello` | `version`, `models` | first, on connecting: the models served so far, as `/v1/models` lists them |
| `model.loading` | `model`, `detail` | a model is loading; `detail` says what it is doing |
| `model.added` | `model` | a model has loaded and is served from now on |
| `model.failed` | `model`, `error` | a model failed to load (the server then stops) |
| `request.started` | `id`, `method`, `path` | an API request arrived (`id` is also its `x-request-id`) |
| `request.finished` | `id`, `method`, `path`, `status`, `duration_ms` | it was answered (for a stream, when the stream started) |
| `heartbeat` | | after 15 s with nothing else to send |
| `lagged` | `missed` | this subscriber fell behind and missed some events |

The server listens before its models load, so a client sees them load and arrive. Events are
not stored: a subscriber gets those published while it is connected. What a client sends is
ignored. `/health` and `/v1/events` aren't reported as requests.

## Getting started

The binary serves the models it is built with. Each model is a feature, with the backend it
runs on:

```bash
cargo install teeny-api --features functiongemma3,cuda
teeny-api --model functiongemma3 --port 8000

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "functiongemma3", "messages": [{"role": "user", "content": "hello"}]}'
```

`--model <tag>` (repeatable) picks the models; the weights come from the local Hugging Face
cache, or `--model-dir` for a single model, and `--context` sets the KV-cache size. `--echo`
serves the echo models of the domains built in (`teeny-echo` for chat, `teeny-echo-complete`
for completions, `teeny-echo-embed` for embeddings, `teeny-echo-moderate` for moderations;
`teeny-echo-transcribe` for transcriptions and `teeny-echo-speak` for speech with `audio`;
`teeny-echo-image` for images with `image`), for trying the API without weights. Each endpoint is answered by the models that
implement its API; a request to a model without it gets a `400` saying what the model does
serve.

The Vue scaffolds in `webapps/` (chat UI, admin console) aren't wired up yet.

## License

Apache-2.0. See [LICENSE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE).
