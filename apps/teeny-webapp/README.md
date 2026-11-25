# teeny-webapp

A browser UI for [teenygrad](https://teenygrad.org)'s
[`teeny-api`](https://docs.teenygrad.org/api/teenygrad/teeny_api/), in one self-contained
binary: the Vue app in `ui/` is built with npm when the crate is, and embedded.

- **Chat** with any model the API serves, streamed: a system prompt, max tokens, temperature,
  and OpenAI-style tool declarations (with a tool choice). The model's tool calls are shown,
  and you type each result for it to carry on.
- **Console**, following the API's `/v1/events` websocket: the models loading (with what they
  are doing), served or failed, and each request as it starts and finishes.

The browser talks to teeny-api directly; this server only serves the app and tells it where
the API is (`/config.json`).

## Getting started

Building needs Node (^20.19 or >=22.12) and npm, for the UI.

```bash
teeny-api --echo                               # or --model functiongemma3, on port 8000
cargo run -p teeny-webapp -- --api-url http://127.0.0.1:8000
# open http://127.0.0.1:8080
```

`--api-url` is where the *browser* reaches teeny-api; `--host` and `--port` (default 8080) are
where this server listens. teeny-api lets any origin call it unless it's given
`--allow-origin http://127.0.0.1:8080` (repeatable).

`TEENY_WEBAPP_UI=skip` builds without Node, embedding a placeholder page instead of the app
(for API docs, say). See [`ui/README.md`](ui/README.md) to work on the app itself.

Not published to crates.io yet: build it from the repository.

## License

Apache-2.0. See [LICENSE](https://github.com/teenygrad/teenygrad/blob/main/LICENSE).
